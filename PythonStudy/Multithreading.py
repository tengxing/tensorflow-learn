############################################################################
# -*- coding:utf8 -*-                                                      #
# created by tengxing on 2017.5.19                                         #
# mail tengxing7452@163.com                                                #
# github github.com/tengxing                                               #
# description python学习 多线程                                               #
############################################################################
import time
import multiprocessing
from multiprocessing import Pool

def do(n) :
  #获取当前线程的名字
  name = multiprocessing.current_process().name
  print name,'starting'
  print "worker ", n
  return

if __name__ == '__main__' :
  numList = []
  for i in xrange(5) :
    #多线程处理目标方法
    '''
    创建子进程时，只需要传入一个执行函数和函数的参数，创建一个Process实例，并用其start()方法启动，这样创建进程比fork()还要简单。
    join()方法表示等待子进程结束以后再继续往下运行，通常用于进程间的同步。
    '''
    p = multiprocessing.Process(target=do, args=(i,))
    numList.append(p)
    p.start() #开始执行
    p.join()
    print "Process end."

  #执行方法
  def run(i):
    time.sleep(1)
    return i*i

  a = [1,1,1,1,1,1]
  start = time.time()
  #单线程
  for i in a:
    run(i)
  end = time.time()
  print "单线程用时:",end-start
  #多线程
  pool = Pool(5)
  rl = pool.map(run,a)
  pool.close()
  pool.join()
  print "多线程用时:",time.time()-end








