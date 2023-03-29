'''
파이썬 기반 수치 해석 라이브러리
선형대수 연산에 필요한 다차원 배열과 배열 연산을 수행하는 다양한 함수 제공

# 설치방법 :  pip install numpy
# 사용방법 : import numpy as np

numpy에서의 배열 ndarray or array라 부름
# numpy.array 
- numpy array 끼리 연산이 가능
ex) [10, 5, 3, 7] + [10, 5, 3, 7] = [20, 10, 6, 14]
- array 전체에 연산이 가능
ex) [10, 5, 3, 7] + 5 = [15, 10, 8, 12]


# python.array
- 배열끼리 덧셈만 가능
ex) [10, 5, 3, 7] + [10, 5, 3, 7] = [10, 5, 3, , 10, 5, 3, 7]
- 배열에 곱셈 (list 요소 반복) 만 가능
ex) [10, 5, 3, 7] * 2 = [10, 5, 3, 7, 10, 5, 3, 7]

# 차원의 크기
shape(4, 3, 2) => x, y, z
ex1) 2개의 데이터 축, 첫번째 축은 길이가 2, 두번째 축은 길이가 3
[
    [1, 0, 0],
    [0, 1, 2]
]
ex2) 1개의 축, 3가지 요소, 길이는 3
[1, 2, 3]

## NumPy 배열 대표 속성값

- ndarray.shape : 배열 각 축의 크기
- ndarray.ndim : 축의 개수 (Dimension)
- ndarray.dtype : 각 요소 (Element)의 타입
- ndarray.itemsize : 각 요소(Element)의 타입의 bytes크기
- ndarray.size : 전체 요소의 개수

ex) .shape = (3, 4)   .ndim = 2        .size = 12      .itemsize = 8 (int64)
[
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11]
]
'''

# import numpy as np

# a = np.arange(12).reshape(3, 4) # a 에 (3, 4) 크기의 2D 배열 생성
# print(a)
# print("\n")
# print(a.dtype) # int32
# print("\n")
# print(a.itemsize) # 4
# print("\n")
# print(a.size) # 12

'''
np.array를 이용하여 Tuple(튜플)이나 List(리스트) 입력으로 numpy.ndarray를 생성

np.zeros(shape) : 0으로 구성된 N차원 배열
np.ones(shape) : 1로 구성된 N차원 배열 생성
np.empty(shape) : 초기화되지 않은 N차 배열 생성
'''

# import numpy as np

# a = np.array([2, 3, 4])
# print(a) # [2, 3, 4]
# print(a.dtype) # int32

# b = np.array([1.2, 3.5, 5.1])
# print(b) # [1.2, 3.5, 5.1]
# print(b.dtype) # float64
# print("\n")

# # (3, 4) 크기의 배열을 생성하여 0으로 채움
# print(np.zeros((3, 4)))
# print("\n")

# # (2, 3, 4) 크기의 배열을 생성하여 1로 채움
# print(np.ones((2, 3, 4))) # 코랩에서는 , dtype=int64 입력 필요
# print("\n")

# # 초기화 되지 않은 (2, 3) 크기의 배열을 생성
# print(np.empty((2, 3)))
# print("\n")

'''
np.arange : N 만큼 차이 나는 숫자 생성
np.linspace : N 등분한 숫자 생성

np.arrange Vs np.linspace
# np.arrange
- [] 생략 가능, 끝 값 포함 X
- 장점 : step, 범위를 구간, 간격 강조할 때 사용하면 코드 가독성 UP !

# np.linspace
- 처음 값과 끝 값 포함, 몇 개로 만들지 매개변수로 줌
- 장점 : 개수 강조할 때 사용하면 코드 가독성 UP!

'''

# import numpy as np

# # 10 이상 30 미만 까지 5씩 차이나게 생성
# print(np.arange(10, 30, 5)) # [10 15 20 25]
# print("\n")

# # 0 이상 2 미만까지 0.3씩 차이나게 생성
# print(np.arange(0, 2, 0.3)) # [0.  0.3 0.6 0.9 1.2 1.5 1.8]
# print("\n")

# # 0 ~ 99 까지 100 등분
# print(np.linspace(0, 99, 100))
# print("\n")

# # arange 함수는 끝 값을 포함하지 않기 때문에
# # 가독성을 위해서 1.25를 1+0.25로 표현!
# # 0부터 1.25 미만까지(끝 값 포함 안하니까 1까지 출력) 0.25씩 차이나게 생성 
# print(np.arange(0, 1 + 0.25, 0.25))
# print("\n")

# # 위와 동일한 결과를 linspace 함수를 이용하여 코딩
# # 0부터 1까지(끝 값 포함) 5등분
# print(np.linspace(0,1,5))
# print("\n")

'''
3차원 배열 출력하는 코드 : (3, 4) 크기의 2차원 배열이 2개 출력되는 형식
c = np.arange(24).reshape(2, 3, 4)
print(c)

'''

# import numpy as np

# # 1차원 배열 출력
# print(np.arange(6))
# print("\n")

# # 2차원 배열 출력
# print(np.arange(12).reshape(4, 3))
# print("\n")

# # 3차원 배열 출력 : 대괄호의 갯수로 판단
# # (3, 4) 크기의 2차원 배열이 2개 출력됨
# print(np.arange(24).reshape(2, 3, 4))
# print("\n")

'''

numpy에서 수치 연산은 기본적으로 element-wise 연산
: 차원(축)을 기준으로 행렬 내에서 같은 위치에 있는 원소끼리 연산을 하는 방식

numpy 여러가지 곱셈 존재
 * : 각각의 원소끼리 곱셈 (Elementwise product, Hadamard product)
 @ : 행렬 곱셈 (Matrix product)

numpy 자동 형 변환
 수치 연산을 진행 할 때 각각의 .dtype이 다르면 타입이 큰 쪽
 ( int < float < complex ) 으로 자동으로 변경

'''

# import numpy as np

# # a와 b 배열 생성하여 출력
# a = np.array([20,30,40,50]) # [20 30 40 50]
# b = np.arange(4) # [0 1 2 3]
# print(a)
# print(b)
# print("\n")

# # a에서 b에 각각의 원소를 - 연산
# c = a-b
# print(c) # [20 29 38 47]
# print("\n")

# # b 각각의 원소에 제곱 연산
# print(b**2) # [0 1 4 9]
# print("\n")

# # a 각각의 원소에 *10 연산
# print(10*a) # [200 300 400 500]
# print("\n")

# # a 각각의 원소가 35보다 작은지 Boolean 결과
# print(a < 35) # [ True  True False False]
# print("\n")

'''
여러가지 곱셈
'''
# import numpy as np

# # A와 B 배열 생성하여 출력
# A = np.array( [[1,1],
#                [0,1]] )
# B = np.array( [[2,0],
#                [3,4]] )
# print(A)
# print(B)
# print("\n")

# # A * B 
# # 각각의 원소끼리 곱셈
# print(A * B)
# print("\n")

# # A @ B
# # 행렬 곱셈 사용
# print(A @ B)
# print("\n")


'''
자동 형 변환
'''
# import numpy as np

# # a와 b 배열 생성 & 타입 확인
# a = np.ones(3, dtype=np.int32)
# b = np.linspace(0, np.pi,3)

# print(a) # [1 1 1]
# print(b) # [0.         1.57079633 3.14159265]
# print(a.dtype) # int32
# print(b.dtype) # float64
# print("\n")

# # a(int), b(float) 연산 시 float로 upcasting
# c = a + b
# print(c) # [1.         2.57079633 4.14159265]
# print(c.dtype) # float64
# print("\n")

# # 마찬가지로 복소수 연산 시 complex(복소수)로 upcasting
# # exp 함수는 지수 함수 
# d = np.exp(c*1j)
# print(d) # [ 0.54030231+0.84147098j -0.84147098+0.54030231j -0.54030231-0.84147098j]
# print(d.dtype) # complex128 : 복소수
# print("\n")

'''
# numpy 집계함수
 .sum : 모든 요소의 합
 .min : 모든 요소 중 최소값
 .max : 모든 요소 중 최대값
 .argmax : 모든 요소 중 최대값의 인덱스
 .cumsum : 모든 요소의 누적합
 
# numpy 집계함수 axis 값을 매개변수로 입력
 축을 기준으로 연산 가능
 axis = 0 (열 기준)
 axis = 1 (행 기준)

'''

# import numpy as np

# # a 배열 생성 & 출력
# # 0부터 8미만까지 출력하고 (2,4) 크기로 재가공하고 제곱하여 출력 
# a = np.arange(8).reshape(2, 4)**2
# print(a)
# print("\n")

# # 모든 요소의 합
# print(a)
# print(a.sum())
# print("\n")

# # 모든 요소 중 최소값
# print(a.min())
# print("\n")

# # 모든 요소 중 최대값
# print(a.max())
# print("\n")

# # 모든 요소 중 최대값의 인덱스 : 배열처럼 0번째 부터 시작
# print(a)
# print(a.argmax())
# print("\n")

# # 모든 요소의 누적합 
# # 14 = 0 + 1 + 4 + 9
# print(a)
# print(a.cumsum())
# print("\n")

# # b 배열 생성 & 출력
# b = np.arange(12).reshape(3, 4)
# print(b)
# print("\n")

# # axis = 0은 열 기준으로 연산
# print(b.sum(axis=0))
# print("\n")

# # axis = 1은 행 기준으로 연산
# print(b.sum(axis=1))
# print("\n")

'''
NumPy 범용 함수
 https://numpy.org/doc/1.18/reference/ufuncs.html#available-ufuncs

'''
# import numpy as np

# # B 배열 생성 & 출력
# B = np.array([1, 4, 9])
# print(B) # [1 4 9]
# print("\n")

# # y = sqrt(x)
# # sqrt는 제곱근 계산
# print(np.sqrt(B))  # [1. 2. 3.]
# print("\n")

'''
Numpy 인덱식(Indexing)과 슬라이싱(Slicing)
 각각 문자열에서 한 개 또는 여러 개를 가리켜서 그 값을 가져오거나 뽑아내는 방법

배열인덱스와 같음 : 0번째부터 시작
ex
 a = np.arange(8)**2
 # [0, 1, 3, 9, 16, 25, 36, 49]
 
 i = np.aray([1, 1, 3, 5])
 a[i] = [1, 1, 9, 25]
 
 j = np.aray([[3, 4], [2, 5]])
 a[j] = [
     [9, 16], 
     [4, 25]
 ]
'''
# import numpy as np

# # a 배열 생성 & 출력
# a = np.arange(10)**2 # [ 0  1  4  9 16 25 36 49 64 81]
# print(a)
# print("\n")

# # a 배열의 2번째 인덱스 출력
# print(a[2]) # 4
# print("\n")

# # a 배열의 2~4번 인덱스 출력
# print(a[2:5]) # [ 4, 9, 16 ]
# print("\n")

# # reverse : 배열의 요소 거꾸로 출력
# print(a[ : :-1]) # [81 64 49 36 25 16  9  4  1  0]
# print("\n")

# # 0~5번에서 2Step 인덱스 출력
# # a[0:6:2] = a[:6:2]
# # 인덱스 0, 2, 4 해당하는 값에 1000 삽입
# a[0:6:2] = 1000
# print(a) # [1000    1 1000    9 1000   25   36   49   64   81]
# print("\n")

'''
인덱스 배열로 인덱싱
'''
# import numpy as np

# # a 배열 생성 & 출력
# a = np.arange(8)**2
# print(a) # [ 0  1  4  9 16 25 36 49]
# print("\n")

# # i 1차원 배열 생성 & 출력
# # a 배열의 index로 i를 삽입하여 출력
# i = np.array([1,1,3,5])
# print(a[i]) # [ 1  1  9 25]
# print("\n")

# # j 2차원 배열 생성 
# # a 배열의 index로 j를 삽입하여 출력
# j = np.array([[3,4], [2,5]])
# print(a[j])
# print("\n")
# # [[ 9 16]
# #  [ 4 25]]

'''
numpy boolean 인덱싱
 boolean 타입을 가진 값들로 인덱싱 진행 : True / False
'''

# import numpy as np

# # a 배열 생성 & 출력
# a = np.arange(12).reshape(3,4)
# print(a)
# print("\n")

# # b는 a > 4 조건이 적용된 Boolean 값이 든 배열
# b = a > 4 
# print(b)
# print("\n")

# # Boolean 값이 든 b 배열을 a 배열의 index로 삽입
# # True인 값들만 출력
# print(a[b])
# a[b].shape
# print("\n")

# # a[b]에 해당하는 애들만 0 삽입하여 a 출력
# a[b] = 0
# print(a)
# print("\n")

'''
# Numpy 크기 변경

np.ndarray의 shape를 다양한 방법으로 변경 가능
 .ravel : 1차원으로 변경
 .reshape : 지정한 차원으로 변경
 .T : 전치(Transpose) 변환
'''
# import numpy as np

# # a 배열 생성 & shape 출력
# a = np.arange(12).reshape(3,4)
# print(a)
# print(a.shape)
# print("\n")

# # .ravel : 모든 원소를 1차원으로 변경
# print(a.ravel())
# print(a.reshape(-1))
# print("\n")

# # .reshape : 지정한 차원으로 변경
# # [3,4] => [2,6]로 변경
# print(a.reshape(2,6))
# print("\n")

# # .T : [3,4]의 전치(transpose)변환으로 [4,3] 출력
# print(a.T)
# print(a.T.shape)
# print("\n")

# # 만약 전치 형태로 a 배열에 저장하고 싶다면! 
# a = a.T
# print(a)
# print("\n")

'''
Numpy 데이터 합치기
 np.vstack : axis = 0 (열) 기준으로 쌓음
 np.hstack : axis = 1 (행) 기준으로 쌓음

'''
# import numpy as np

# # a 배열 생성 & 출력 
# a = np.array([1, 2, 3, 4]).reshape(2, 2)
# print(a)
# print("\n")

# # b 배열 생성 & 출력
# b = np.array([5, 6, 7, 8]).reshape(2, 2)
# print(b)
# print("\n")

# # [2,2] => [4,2]
# # np.vstack(): axis=0(열) 기준으로 쌓음
# print(np.vstack((a,b)))
# print("\n")

# # [2,2] => [2,4]
# # np.hstack(): axis=1(행) 기준으로 쌓음
# print(np.hstack((a,b)))
# print("\n")

'''
Numpy 데이터 쪼개기
 np.hsplit 통해 숫자 1개가 들어갈 경우, X개로 등분
 np.hsplit 통해 리스트로 넣을 경우, axis = 1 (행) 기준 인덱스로 데이터를 분할
'''
# import numpy as np

# # a 배열 생성 & 출력 
# a = np.arange(12).reshape(2, 6)
# print(a)
# print("\n")

# # [2,6] => [2,2] 데이터 3개로 등분
# print(np.hsplit(a, 3))
# print("\n")

# # [2,6] => [:, :3], [:, 3:4], [:, 4:]로 분할
# # a를 3번째 열 ~ 4번째 열 미만 기준으로 분할하여 3개의 array를 반환
# print(a)
# print(np.hsplit(a, (3,4)))
# print("\n")