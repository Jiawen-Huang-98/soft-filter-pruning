# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：add_argument.py
@时间：2022/3/2  15:03
@文档说明:
"""
import argparse
import math

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
#
# args = parser.parse_args()
# print(args.accumulate(args.integers))


# parser = argparse.ArgumentParser(prog = 'PROG')
# parser.add_argument('--foo',nargs = '?',help = 'foo help')
# parser.add_argument('bar',nargs = '+',help = 'bar help')
# parser.print_help()


# 当narg = N (一个整数)表示将命令行中的N个参数聚集到一个列表中
# parser = argparse.ArgumentParser()
# parser.add_argument('--foo', nargs=2)
# parser.add_argument('bar', nargs=1)
# parser.parse_args('c --foo a b'.split())


# parser=argparse.ArgumentParser(prog = 'PROG')
# parser.add_argument('--foo')
# parser.add_argument('command')
# parser.add_argument('args',nargs = argparse.REMAINDER)
# print(parser.parse_args('--foo B cmd boss XX ZZ'.split()))


# parser=argparse.ArgumentParser()
# parser.add_argument('foo',type=int)
# parser.add_argument('bar',type = open)  #此处的open是open()函数，用处是打开文件
# parser.parse_args('2 temp.txt'.split())
# print(parser.parse_args())


# parser = argparse.ArgumentParser()
# parser.add_argument('bar',type = argparse.FileType('w'))
# parser.parse_args(['out.txt'])
# print(parser.parse_args())


# def perfect_square(string):
#     value = int(string)
#     sqrt = math.sqrt(value)
#     if sqrt != int(sqrt):
#         msg = "%r is not a perfect square" % string
#         raise argparse.ArgumentTypeError(msg)
#     return value
#
#
# parser = argparse.ArgumentParser(prog='PROG')
# parser.add_argument('foo', type=perfect_square)
# # parser.parse_args(['9'])
# print(parser.parse_args(['9']))
# # parser.parse_args(['7'])
# print(parser.parse_args(['7']))


parser = argparse.ArgumentParser()
parser.add_argument('--foo')
parser.add_argument('bar')
# parser.parse_args('X --foo Y'.split())

parser.print_help()