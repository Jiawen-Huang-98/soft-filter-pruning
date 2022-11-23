# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：class.py
@时间：2022/3/1  21:02
@文档说明:
"""


class Cat:
    # 初始化方法的参数，创建类时，必须给足
    def __init__(self,name,sex):
        print('初始化方法'),
        self.name = name
        self.sex = sex


# 使用类创建一个对象时会调用初始化方法
cat = Cat("lala",'公')
print(cat.name,cat.sex)

dog = Cat('dog','母')
print(dog.name,dog.sex)






