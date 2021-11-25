# RICKY

## Python上下文管理协议:__enter__和__exit__

代码示例:

```python
 1 class Sample:
 2     def __enter__(self): 
 3         print "In __enter__()" 
 4         return "Foo"
 5 
 6      def __exit__(self, type, value, trace):
 7          print "In __exit__()"
 8 
 9     def get_sample(): 
10         return Sample() 
11 
12 with get_sample() as sample: 
13     print "sample:", sample 
```

输出如下：

 In __enter__()
 sample: Foo
 In __exit__()

## python之抽象基类abc.abstractmethod

有时，我们抽象出一个基类，知道要有哪些方法，但只是抽象方法，并不实现功能，只能继承，而不能被实例化，但子类必须要实现该方法，这就需要用到抽象基类。

```python
@abc.abstractmethod
def get_jacobi(self, parent):
    """abstract function"""
```

## assert断言

在表达式条件为false的时候触发异常。

![img](https://www.runoob.com/wp-content/uploads/2019/07/assert.png)

## numpy常用操作

