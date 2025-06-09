import unittest

def load_test_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    from . import test_inference, test_metrics, test_patterns, test_performance
    
    suite.addTests(loader.loadTestsFromModule(test_inference))
    suite.addTests(loader.loadTestsFromModule(test_metrics))
    suite.addTests(loader.loadTestsFromModule(test_patterns))
    suite.addTests(loader.loadTestsFromModule(test_performance))
    
    return suite

def run_all_tests():
    suite = load_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite) 