from abc import abstractmethod

import torch
from deeplite.profiler.evaluate import EvaluationFunction
from deeplite.profiler.utils import Device, cast_tuple


def cudafy(*args, **kwargs):
    return funcify('cuda', args, **kwargs)


def cpufy(*args, **kwargs):
    return funcify('cpu', args, **kwargs)


def itemify(*args):
    return funcify('item', args)


def _itemwise_funcify(x, name, be_nice, **kwargs):
    if be_nice and not hasattr(x, name):
        return x
    return getattr(x, name)(**kwargs)


def funcify(func, items, be_nice=False, keep_iterable=False, **kwargs):
    if isinstance(func, str):
        name = func
        def func(x):
            return _itemwise_funcify(x, name, be_nice, **kwargs)

    rval = list(map(func, items))
    if not keep_iterable and len(rval) == 1:
        return rval[0]
    return rval


class TorchEvaluationFunction(EvaluationFunction):
    """
    Does the common groundwork for every torch model at evaluation:
        * sending to eval
        * calling torch.no_grad()
        * cpu / gpu device calls

    NOTE: Since we now simply rely on the implementation of _compute_inference, concrete implementations
    cannot ask for more keywords through __call__ (if they want to respect the profiler's signature
    limitation). See _GetTopk implementation for the little hack work around.
    """

    def __call__(self, model, data_loader, device=Device.CPU, transform=None):
        if device == Device.GPU:
            cudafy(model)
        else:
            cpufy(model)
        model = model.eval()
        with torch.no_grad():
            rval = self._compute_inference(model, data_loader, device=device, transform=transform)
        return rval

    @abstractmethod
    def _compute_inference(self, model, data_loader, device=Device.CPU, transform=None):
        raise NotImplementedError("Base class call")


class _GetMissclass(TorchEvaluationFunction):
    def _compute_inference(self, model, data_loader, device=Device.CPU, transform=None):
        total_acc = 0
        for x, y in data_loader:
            if len(x.shape) == 3:
                x = x[None]
            if transform:
                x, y = transform(x, y)
            if device == Device.GPU:
                x, y = cudafy(x, y)
            out = model(x)

            if out.shape[1] == 1:
                out = torch.gt(out, 0)
            else:
                out = torch.nn.functional.softmax(out, dim=-1)
                out = torch.argmax(out, dim=1)

            if out.dim() == 1 and y.dim() == 2 and y.shape[1] == 1:
                y = y.flatten()

            acc = torch.mean((out != y).float())
            total_acc += itemify(acc)
        return 100. * (total_acc / float(len(data_loader)))
get_missclass = _GetMissclass()


class _GetAccuracy(_GetMissclass):
    def _compute_inference(self, model, data_loader, device=Device.CPU, transform=None):
        rval = super()._compute_inference(model, data_loader, device, transform)
        return 100. - rval
get_accuracy = _GetAccuracy()


class _GetTopk(TorchEvaluationFunction):
    def __init__(self):
        self.__topk = None

    def __call__(self, model, data_loader, device=Device.CPU, topk=(1, 5), transform=None):
        topk = cast_tuple(topk)
        self.__topk = topk
        rval = super().__call__(model, data_loader, device, transform)
        self.__topk = None
        return {'top-' + str(k): res for k, res in zip(topk, rval)}

    def _compute_inference(self, model, data_loader, device=Device.CPU, transform=None):
        topk = self.__topk

        def _accuracy(output, target):
            """Computes the precision@k for the specified values of k"""
            maxk = max(topk)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()

            correct = pred.eq(target.view(1, -1))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k)
            return res

        correct_k = torch.FloatTensor([0.] * len(topk))
        ntotal = 0
        for xs, ys in data_loader:
            if len(xs.shape) == 3:
                xs = xs[None]
            if transform:
                xs, ys = transform(xs, ys)
            if device == Device.GPU:
                xs, ys = cudafy(xs, ys)
            outputs = model(xs)
            correct_k += torch.cat(_accuracy(outputs, ys), 0).cpu()
            ntotal += ys.size(0)
        return (100. * (correct_k / ntotal)).tolist()
get_topk = _GetTopk()


class EvalLossFunction(TorchEvaluationFunction):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def _compute_inference(self, model, data_loader, device=Device.CPU, transform=None):
        self.loss_fn.to_device(device)

        loss = 0
        for batch in data_loader:
            if transform:
                batch = transform(batch)
            loss_ = self.loss_fn(model, batch)
            if isinstance(loss_, dict):
                loss_ = sum(loss_.values())
            loss += loss_.item()
        return loss / len(data_loader)
