# imitative-learning-car-racing

## Car Racing repository:

https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py

## bibliography 

https://towardsdatascience.com/beginners-guide-to-loading-image-data-with-pytorch-289c60b7afec

https://www.learnpytorch.io/pytorch_most_common_errors/

## Libraries:

1. gym (<code>pip install gym</code>)
2. pickle-mixin (<code>pip install pickle-mixin</code>)
3. torch (<code>pip install torch</code>)
4. Box2D (<code>pip install Box2D</code>)
5. Box2D-kengz (<code>pip install box2d-kengz</code>)
6. numpy (<code>pip install numpy</code>)
7. matplotlib (<code>pip install matplotlib</code>)
8. torchvision (<code>pip install torchvision</code>)

Or just run <code>pip install -r requirements.txt</code>

OBS: Box2D requires swig, so be aware of that.

## Usage

```bash
python ./src/ag.py <epochs> <batch_size> <track> <should-train>
```

### Default Values

epochs = 100

batch_size = 1

track = 42

should-train = 1 (True)

OBS: Track must be an integer between 1 and 100

OBS 2: should-train is a boolean, should be 0 (False) or 1 (True)

This software will primarily look for a CUDA-enabled device. If it's unable to find one, it will use your CPU instead.

## Todo list:

- [x] Install libraries
- [x] Copy code "car_racing.py"
- [ ] 
- [ ] 
- [ ]
- [ ] 
- [ ]
- [ ] 
- [ ] 
