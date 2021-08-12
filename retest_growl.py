from subprocess import Popen
import sys
import time

filename = 'growl_tests_code.py'
iter = 0
while True:
    print("\nStarting " + filename)
    p = Popen("python " + filename, shell=True)
    p.wait()
    iter += 1
    time.sleep(60)
    print('DONE ', iter, ' STEPS')
    if iter == 30:
        break