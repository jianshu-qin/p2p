# enviroment
The code should only requires diffusers==0.18.2.My enviroment is same as https://github.com/haohantianchen/Animate.

# prompt to prompt edit
```
python test_p2p.py
```
you can modify the prompts in the code

# null-text inversion

## reconstruct a real image
```
python test_nti.py
```
You may need to change the image path and prompts.
To abtain more accurate reconstrution results, increase the `inner_steps` args.
More detail can be seen by running `python test_nti.py --help`

## edit the reconstructed image by p2p
```
python test_nti.py --inverse False --edit True
```
The code is still has bug: background is changed. I'm working to solve it.


# warp fg and blend bg in image level
```
python test_p2p_c.py
```