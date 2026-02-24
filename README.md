**SCOPE and REACH Optimized Inference Package**
This repository contains the code for a Python package that aims to provide a general and easy to interface with implementation of the SCOPE and REACH estimators. 

All that is required to begin creating generative risk scores is:

1. An SGLang compatible model
2. Tokenized sequences presented as a list of a list of ints
3. The outcome of interest token id
4. Suppressed token ids (elements in your vocab like padding tokens that you don't want your model generating)
5. Stop token ids (The tokens that signal the end of a timeline)

There is also WIP support for time based termination. Currently this only allows for time spacing tokens. Future additions will make this more general and hopefully even allow users to define their own termination logic adhoc. 
