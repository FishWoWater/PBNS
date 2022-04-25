**A fork version of PBNS to support Resizer for arbitrary poses(see [PBNS_Resizer](PBNS_Resizer/README.md) for more details) **



PBNS: Physically Based Neural Simulation for Unsupervised Outfit Pose Space Deformation.





This repository contains the necessary code to run the model described in:<br>
https://arxiv.org/abs/2012.11310

<img src="https://sergioescalera.com/wp-content/uploads/2021/01/clothed31.png">

<img src="/gifs/seqs.gif">

Video:<br>
https://youtu.be/ALwhjm40zRg

<h3>Outfit resizing</h3>

PBNS formulation also allows unsupervised outfit resizing. That is, retargetting to the desired body shape and control over tightness.<br>
Just as standard PBNS, it can deal with complete outfits with multiple layers of cloth, different fabrics, complements, ...

<p float='left'>
  <img width=400px src="/gifs/resizer0.gif">
  <img width=400px src="/gifs/resizer1.gif">
</p>

<h3>Enhancing 3D Avatars</h3>

Due to the simple formulation of PBNS and no dependency from data, it can be used to easily enhance any 3D custom avatar with realistic outfits in a matter of minutes!

<p float='left'>
  <img width=300px src="/gifs/avatar1.gif">
  <img width=300px src="/gifs/avatar2.gif">
  <img width=300px src="/gifs/avatar0.gif">
</p>

<h3>Repository structure</h3>
This repository is split into two folders. One is the standard PBNS for outfit animation. The other contains the code for PBNS as a resizer.<br>
Within each folder, you will find instructions on how to use each model.
