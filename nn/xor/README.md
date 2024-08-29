# XOR Neural Network

Network predicting the outcome of an XOR operation.

<br>

## Example

```
A B | XOR
0 0 | 0
0 1 | 1
1 0 | 1
1 1 | 0
```

<br>

## Network Structure

- Input Layer: 2 Neurons
- Hidden Layer: 1 Layer with 2 neurons
- Output Layer: 1 Neuron

<br>

## Output

```
Epoch: 0
Input: 0 0 | Expected Output: 0 | Output: 0.708853
Input: 0 1 | Expected Output: 1 | Output: 0.744938
Input: 1 1 | Expected Output: 0 | Output: 0.774149
Input: 1 0 | Expected Output: 1 | Output: 0.740061

Epoch: 1000
Input: 0 0 | Expected Output: 0 | Output: 0.441946
Input: 0 1 | Expected Output: 1 | Output: 0.512032
Input: 1 1 | Expected Output: 0 | Output: 0.54771
Input: 1 0 | Expected Output: 1 | Output: 0.522746

Epoch: 2000
Input: 0 1 | Expected Output: 1 | Output: 0.580212
Input: 1 0 | Expected Output: 1 | Output: 0.623989
Input: 0 0 | Expected Output: 0 | Output: 0.2896
Input: 1 1 | Expected Output: 0 | Output: 0.596346

Epoch: 3000
Input: 0 0 | Expected Output: 0 | Output: 0.209211
Input: 1 1 | Expected Output: 0 | Output: 0.379437
Input: 0 1 | Expected Output: 1 | Output: 0.704199
Input: 1 0 | Expected Output: 1 | Output: 0.71153

Epoch: 4000
Input: 1 0 | Expected Output: 1 | Output: 0.852246
Input: 0 0 | Expected Output: 0 | Output: 0.13827
Input: 0 1 | Expected Output: 1 | Output: 0.852925
Input: 1 1 | Expected Output: 0 | Output: 0.173888

Epoch: 5000
Input: 1 1 | Expected Output: 0 | Output: 0.11223
Input: 0 1 | Expected Output: 1 | Output: 0.899567
Input: 0 0 | Expected Output: 0 | Output: 0.101351
Input: 1 0 | Expected Output: 1 | Output: 0.899467

Epoch: 6000
Input: 1 1 | Expected Output: 0 | Output: 0.0863917
Input: 0 1 | Expected Output: 1 | Output: 0.921302
Input: 0 0 | Expected Output: 0 | Output: 0.0823796
Input: 1 0 | Expected Output: 1 | Output: 0.921197

Epoch: 7000
Input: 1 1 | Expected Output: 0 | Output: 0.0718635
Input: 1 0 | Expected Output: 1 | Output: 0.933727
Input: 0 1 | Expected Output: 1 | Output: 0.933924
Input: 0 0 | Expected Output: 0 | Output: 0.0707444

Epoch: 8000
Input: 0 0 | Expected Output: 0 | Output: 0.0627033
Input: 1 1 | Expected Output: 0 | Output: 0.0623745
Input: 0 1 | Expected Output: 1 | Output: 0.942107
Input: 1 0 | Expected Output: 1 | Output: 0.942052

Epoch: 9000
Input: 0 1 | Expected Output: 1 | Output: 0.948134
Input: 1 0 | Expected Output: 1 | Output: 0.948081
Input: 1 1 | Expected Output: 0 | Output: 0.0557886
Input: 0 0 | Expected Output: 0 | Output: 0.0568253

Epoch: 10000
Input: 1 1 | Expected Output: 0 | Output: 0.0506308
Input: 0 1 | Expected Output: 1 | Output: 0.952592
Input: 1 0 | Expected Output: 1 | Output: 0.952543
Input: 0 0 | Expected Output: 0 | Output: 0.0522638


Final Hidden Weights
[
  [ 5.798517 5.824547 ]
  [ 3.852110 3.857878 ]
]

Final Hidden Biases
[ -2.453169 -5.910647 ]

Final Output Weights
[ 7.686895 -8.312762 ]

Final Output Biases
[ -3.484501 ]

```
