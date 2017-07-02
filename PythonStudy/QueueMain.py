############################################################################
# -*- coding:utf8 -*-                                                      #
# created by tengxing on 2017.5.19                                         #
# mail tengxing7452@163.com                                                #
# github github.com/tengxing                                               #
# description python学习 队列                                               #
############################################################################
import Queue
import sys
reload(sys)
sys.setdefaultencoding('utf8')


if __name__ == '__main__':
    input = Queue.Queue(10)
    input.put("1")
    input.put("2")
    input.put("3")
    print input.get()
    #1
    print input.get()
    #2
    print input.get()
    #3
"""
队列中常用的方法
Queue.qsize() 返回队列的大小
Queue.empty() 如果队列为空，返回True,反之False
Queue.full() 如果队列满了，返回True,反之False
Queue.get([block[, timeout]]) 获取队列，timeout等待时间
Queue.get_nowait() 相当Queue.get(False)
非阻塞 Queue.put(item) 写入队列，timeout等待时间
Queue.put_nowait(item) 相当Queue.put(item, False)


"""