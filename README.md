# MobileMind

My research into machine learning.

*Note: I'm no expert in machine learning. I would not expect this to make your phone be able to train thousands of photos on your phone anytime soon*

## Description
Nothing here is particularly advanced, I just rewrote a basic neural network with 2 inputs and two outputs. I wrote this model based off this [post](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/). I did this to break my viewpoint that machine learning was just magic. Its just a ton of math...

### Why not use BNNS and Metal?
Currently BNNS and Metal don't provide a way to train the model on iOS (for good reason). But since I plan on using rather simple models that won't be trying to train massive amounts of data at the same time, I thought an iPhone would suffice.

## Networks
Right now I have two networks written up, BasicNetwork and BetterNetwork. These two networks are both a copy of this model

![model](https://matthewmazur.files.wordpress.com/2018/03/neural_network-9.png)

### BasicNetwork
BasicNetwork is full write out of all the math done from the post. Its horribly unoptimized and everything is done in a single function. The reason this exists is because I wrote this out to help understand all the math behind it.

I've decided to include it for anyone thats new to machine learning can read it over to get a better understanding of the math and see the expected outputs of the equations with the default values.

#### Usage
```let network = BasicNetwork()
network.train() // does as many passes as it needs to get the output with a 0.005% error
network.apply() // returns the output of the network as a tuple
```

### BetterNetwork
This is a rewritten version of BasicNetwork thats been cleaned up and somewhat optimized. It treats most of the values as vectors and matrixes. This allows the network to use SIMD to compute multiple equations at once.

#### Usage
```let network = BetterNetwork()
network.train() // does as many passes as it needs to get the output with a 0.005% error
network.apply() // returns the output of the network as a 2D float
```

## Limitations
Right now there is no expandability with this code. This is just a proof of concept. Overtime I'll hopefully be able to build out the functionailty to allow a larger number of inputs and layers.

At the moment this code only offers back propagation as a training method. I plan on trying to implement other methods like the [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm).

Currently, the only usable activation function for back propagation is the sigmoid function. This can easily be fixed by finding the derivative of the other functions available (ReLu, tanh)

## Resources
- Matt Mazur's [Blog Post](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
  - I suggest grabbing a pen and paper and read this blog post to wrap your head around the math first and then looking at the code here for help implementing it as code

- Jabrils' Machine Learning Game series (4 Parts) [Playlist](https://www.youtube.com/watch?v=ZX2Hyu5WoFg&list=PL0nQ4vmdWaA0mzW4zPffYnaRzzO7ZqDZ0)
  - I found these videos very informative and helped get a visual understanding how neural networks worked.

- [GitBook](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/neural_networks.html)
  - This website uses an old deprecated CDN of MathJax so the equations are not visible, which will make things confusing. If you are using Chrome I suggest downloading the [Tampermonkey](https://chrome.google.com/webstore/detail/tampermonkey/dhdgffkkebhmkfjojejmpbldmpobfkfo?hl=en) addon and adding this script

```// ==UserScript==
// @name         MathJax CDN
// @namespace    https://pheztech.com/
// @version      0.1
// @description  Use MathJax's new CDN
// @author       KenP
// @match        https://leonardoaraujosantos.gitbooks.io/
// @require      https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js
// @grant        none
// ==/UserScript==
```
This will force the website to use working MathJax CDN so you can see the equations. Since this script is super simple, when you change pages you will have to refresh the page for the equations to become visable.

## License
[GNU GPLv3](../blob/master/LICENSE)
