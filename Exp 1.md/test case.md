TEST CASE1:
CODE:

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected = [0, 1, 1, 0]

print("XOR Problem Test:")
for x, y in zip(inputs, expected):
    print(f"Input: {x}, Expected: {y}")

OUTPUT:
https://github.com/student3020srm-design/Deep-learning/issues/1#issue-3316856818


TEST CASE 2:
def perceptron_predict(x, weights, bias):
  
    total = sum(w * xi for w, xi in zip(weights, x)) + bias  
    return 1 if total >= 0 else 0

weights = [1, 1]  
bias = -0.5

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected = [0, 1, 1, 0]

print("\nPerceptron Prediction Test:")
for x in inputs:
    pred = perceptron_predict(x, weights, bias)
    print(f"Input: {x}, Perceptron Output: {pred}")

OUTPUT:
<img width="878" height="597" alt="Screenshot 2025-08-06 114827" src="https://github.com/user-attachments/assets/e0f48463-6a83-482f-b9f0-2b8d299aed39" />
