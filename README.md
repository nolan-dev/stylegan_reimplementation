This is a reimplementation of NVidia's stylegan https://github.com/NVlabs/stylegan I did for learning purposes.  My priority was testing out changes such as non-square images and conditioning on labels (using acgan and projection discriminator) rather than keeping the code clean so right now it is a bit messy.

This has been tested, and supports features like rectangular images, but I am currently waiting on the most recent training session to finish before uploading results. Training and sample generation are both performed using 'run.py'.  

It is mostly original, but includes a couple functions from the official implementation for comparison testing.

To run, see the comments in start_training.sh (and then run start_training.sh).  This has only been tested in one environment, so feel free to create a github issue if you encounter a problem.  This model can handle non-square images and I plan on organizing and including the tools I've made/modified to generate datasets like that soonish.

Original Paper: 
Karras, T., Laine, S., and Aila, T.
A style-based generator architecture for generative adversarial networks
