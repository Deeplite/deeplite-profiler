import hashlib
import os
from cryptography.fernet import Fernet
import base64

def generate_checksum(fname, block_size=4096):
    hash = hashlib.md5()
    with open(fname, "rb") as f:
        block = f.read(block_size)
        while block:
            hash.update(block)
            block = f.read(block_size)
    return hash.hexdigest()


deeplite_key = os.getenv("DEEPLITE_KEY", "12345681234568123456812345681234")
print("Deeplite key {}".format(deeplite_key))
print("DLTEST {}".format(os.getenv("DLTEST")))

def generate_signature(package_root):
   sums = []
   build_path = None
   for path, dirs, files in os.walk(package_root):
      for filename in files:
         if filename.endswith(".so"):
            sums.append(generate_checksum(os.path.join(path,filename)))
            if not build_path:
               build_path = path
               build_path = build_path.split("deeplite", 1)[0]

   signature =  ",".join(sums)
   f = Fernet(base64.urlsafe_b64encode(deeplite_key.encode("utf-8")))
   signature = f.encrypt(signature.encode("utf-8"))

   return signature.decode('utf-8'), build_path




def write_signature(sig, build_path):
   sig_file_name = os.path.join(build_path,"deeplite/deeplite-profiler.sig")
   sig_file = open(sig_file_name, "w")
   n = sig_file.write(sig)
   sig_file.close()   

if __name__ == "__main__":
   write_signature(*generate_signature("build"))