
# This is for comments

# 파이손 기본 데이터 타입
a = 1 # Number
print(a,type(a))
a = 1.1 # Number
print(a,type(a))
a = True # Boolean
print(a,type(a))
a = "Love" # String
print(a,type(a))

# 숫자 타입에 대한 기본 연산
a, b = 1, 3.4
c = a+b
print(c,type(c))
a, b = 10, 3
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a%b)
print(a**b)

# 논리 타입에 대한 기본 연산
a, b = 1, 3
d = a>b
print(d,type(d))
print(not d)
d = (a>b) & (b==3)
print(d)
d = (a>b) | (b==3)
print(d)

# 연습문제
# 어떤 정수 a가 양의 한자리 수이거나 양의 세자리 수이면 True가 되도록 하는 코드를 작성하시오
# a = 3, 387 --> True,  a = -5, 27, 4357 --> False 



# 파이손에서의 배열

# List 타입
a = [1,2,3]
print(a,type(a))
a = ['Love', True, 1.1, [2,3] ]
print(a,type(a))

# List 타입에 대한 인덱싱
a = ['Love', True, 1.1, 4 ]
a[0]  # 첫번째 요소
a[1]  # 두번째 요소
a[-1] # 마지막 요소
a[-2] # 마지막에서 두번째 요소
a[0:2] # 인덱스 0, 1에 대한 요소들
a[2:] # 인텍스 2에서부터 끝까지
a[:3] # 처음부터 인덱스 2까지
a[:-1] # 처음부터 마지막에서 두번째 까지

# List 타입에 대한 연산
a = [1,2,3]
b = [4,5,6]
a+b
a*2
len(a)
len(a+b)
a[1] = "Love"


# Tuple type
# List와 비슷하지만 변경이 불가
a = (1,2,3)
print(a,type(a))
a[1] = "Love"

a = [1]
b = (1)
c = (1,)

# 연습문제
# a = [1, 2, 3, [4,5] ] 에 대하여 5를 6으로 바꾸시오. 



# 파이손에서의 컨트롤

# 조건문
score = 80
if score>75:
    print("pass")
else:
    print("fail")

if score>90:
    print("A")
elif score>80:
    print("B")
else:
    print("F")


# 반복문
for x in [1,2,3,4]:
    print(x**2)

a = [1,2,3,4]
for i in range(4):
    print(a[i]**2)

s = 0
for x in range(11):
    s = s+x
print(s)

s = 0
for x in range(1,11,2):
    s = s+x
print(s)

s = 0
for x in range(11):
    if x%2 == 1:
        s = s+x
print(s)


# 연습문제
# 1326의 모든 약수를 구하시오 
a = 1326

div = []
for d in range(1,1327):
    if a % d == 0:
        div.append(d)
        print(d)


# NumPy
import numpy

# numpy array
a = [1,2,3,4]
b = numpy.array([1,2,3,4])
a.mean()
b.mean()
b.var()

# numpy in short
import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])
a+b

# indexing: list 인덱싱과 동일
a = np.array([1,2,3,4])


b = np.array([[1, 2, 3], [5, 6, 7]])

# 몇 가지 유용한 함수들
np.arange(10)
np.arange(1,10,2)
np.linspace(0,1,11)
np.ones(5)*2
np.random.rand(10)  # 균등분포에서 난수 생성
np.random.randn(10) # 표준정규분포에서 난수 생성
np.random.seed(1) # random seed


# 연습문제
# 1. 1과 3사이에서 균등분포로 난수 100개를 발생시키시오. 이들을 평균은 얼마인가?
# 2. 위의 난수 중에서 2보다 작은 것은 모두 몇 개인지 세시오. 
Btwn1and3 = np.random.rand(100) * 2 + 1
print(Btwn1and3)
print(Btwn1and3.mean())

print('---------------')
length = len([b for b in Btwn1and3 if b < 2])
(Btwn1and3<2).sum()

count = 0
for r in Btwn1and3:
    if r < 2:
        count += 1
print (length, count, length == count)
