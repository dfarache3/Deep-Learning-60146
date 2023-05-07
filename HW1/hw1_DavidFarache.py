# -*- coding: utf-8 -*-
"""ObjectOrientedHW1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CAjivkC_WSFZvas3UhmrhBrTrfwwJpq2

Programming Task 1
"""

#Create Sequence Class
class Sequence(object):
  def __init__(self, array):
    self.array = array

"""Programming Task 2"""

class Fibonacci(Sequence):
    def __init__(self, first_value, second_value):
      self.first_value = first_value
      self.second_value = second_value

    def __call__(self, length):
      hold_array = []
      #set array to first two values
      hold_array = [self.first_value, self.second_value]
      #calc fib sequence
      for i in range(0, length-2):
        val = hold_array[i] + hold_array[i+1]
        hold_array.append(val)
      self.array = hold_array
      return self.array

FS = Fibonacci ( first_value =1 , second_value =2 )
FS ( length =5 )

"""Programming Task 3"""

class Sequence(object):
  def __init__(self, array):
    self.array = array

  def get_number(self, pos):
    return self.array[pos]

  #return element of array when looped through
  def __iter__(self):
      return (n for n in self.array)

class Fibonacci(Sequence):
    def __init__(self, first_value, second_value):
      self.first_value = first_value
      self.second_value = second_value

    def __call__(self, length):
      hold_array = []
      #set array to first two values
      hold_array = [self.first_value, self.second_value]
      #calc fib sequence
      for i in range(0, length-2):
        val = hold_array[i] + hold_array[i+1]
        hold_array.append(val)
      self.array = hold_array
      return self.array
    
    def __len__(self):
      return len(self.array)

FS = Fibonacci ( first_value =1 , second_value =2 )
print(FS ( length =5 ))
print(len( FS )) 
print([n for n in FS])

"""Porgramming Task 4"""

class Prime(Sequence):
    def __init__(self):
      self.value = 2

    def __call__(self, length):
      self.value = 2
      self.array = [2]
      #calc prime number
      while len(self.array) < length:
        self.value = self.value + 1
        flag = True

        for i in self.array:
          if(self.value % i == 0):
            flag = False

        if(flag == True):
          self.array.append(self.value)

      return self.array
    
    def __len__(self):
      return len(self.array)

PS = Prime ()
PS ( length =8 )
print(PS ( length =8 ))
print (len( PS ) )
print ([n for n in PS])

"""Programming Task 5"""

class Sequence(object):
  def __init__(self, array):
    self.array = array

  def get_number(self, pos):
    return self.array[pos]

  def __iter__(self):
    return (n for n in self.array)

  #Add error message and compare value of arrays
  def __gt__(self, other):
    if(len(self.array) != len(other.array)):
      raise ValueError("Two arrays are not equal in length!")
    
    num_greater = 0
    for i, j in zip(self.array, other.array):
      if(i > j):
        num_greater += 1
        
    return num_greater


class Fibonacci(Sequence):
    def __init__(self, first_value, second_value):
      self.first_value = first_value
      self.second_value = second_value

    def __call__(self, length):
      hold_array = []
      hold_array = [self.first_value, self.second_value]
      for i in range(0, length-2):
        val = hold_array[i] + hold_array[i+1]
        hold_array.append(val)
      self.array = hold_array
      return self.array
    
    def __len__(self):
      return len(self.array)

class Prime(Sequence):
    def __init__(self):
      self.value = 2

    def __call__(self, length):
      self.value = 2
      self.array = [2]
      while len(self.array) < length:
        self.value = self.value + 1
        flag = True

        for i in self.array:
          if(self.value % i == 0):
            flag = False

        if(flag == True):
          self.array.append(self.value)

      return self.array
    
    def __len__(self):
      return len(self.array)

FS = Fibonacci ( first_value =1 , second_value =2 )
FS ( length =8 ) 
PS = Prime ()
PS ( length =8 )
print(FS ( length =8 ) )
print(PS ( length =8 ))
print( FS > PS )
PS ( length =5 ) 
print( FS > PS )

