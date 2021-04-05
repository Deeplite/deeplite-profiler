
class BaseUnitTest:

	def setup_method(self, method):
		""" setup any state tied to the execution of the given method in a
		class.  setup_method is invoked for every test method of a class.
		"""
		print("setup ", method.__name__)
		pass


	def teardown_method(self, method):
		""" teardown any state that was previously setup with a setup_method
		call.
		"""
		print("teardown ", method.__name__)
		pass