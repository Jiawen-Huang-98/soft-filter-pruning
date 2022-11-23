# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：__dict__属性.py
@时间：2022/2/24  15:07
@文档说明:
"""
# import torchvision.models as models
#
# print(models.__dict__['__package__'])
#
# def func():
#     pass

# func.temp = 1
# func.__name__ = '1'
#
# print(func.__dict__)
#
# class TempClass:
#     a = 1
#     def temp_function(self):
#         pass
#
# print(TempClass.__dict__)
#
# class Javen:
#     b=1
#
# print(Javen.__dict__)



# 可通过调试查看变量的结构
import models
model_names = sorted(name for name in models.__dict__   # __dict__是一个字典，其包含了该对象上的全部属性
    if name.islower() and not name.startswith("__")     # islower（）函数验证字符串是否有大写字母，如果是会返回True，startswith()字符串是否以什么什么开头
    and callable(models.__dict__[name]))
dict = models.__dict__
print('model_name:',model_names)
print('models.__dict__:',models.__dict__)