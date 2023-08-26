# Optimization & Vectorization 22/23 course assignments

You can find a **assignment.pdf** document detailing my algorithms and the corresponding implementations in each assignment's folder.

<h3>1) <a target="_blank" rel="noopener noreferrer" href="">Draw the Line</a></h3>
<img align="left" src="https://raw.githubusercontent.com/gianmarcopicarella/ov-assignments-uu/main/readme/draw_the_line.png?token=GHSAT0AAAAAACGZC3RVRMSUL5A3GHQOO4NUZHKIBWQ" width="330">
&nbsp;
<p align="justify">For the purpose of this assignment, you are asked to optimize line rendering alone; 
you can thusleave all other code unmodified. Please do not use the GPU to speed up line rendering, 
and do not use multi-threading. My Implementation is around 5x faster than the baseline code. </p>
<br clear="left"/>

---

<h3>2) <a target="_blank" rel="noopener noreferrer" href="">Velocity Vector</a></h3>
<img align="left" src="" width="330">
&nbsp;
<p align="justify"> The code for this assignment implements a basic cloth simulation on a 256x256 grid using a technique known as Verlet integration.
The goal of this assignment is to improve the performance of the application using vectorization. This can be accomplished with two main technologies discussed during the course: SIMD and GPGPU. My Implementation is around 2.1x and 466x faster using SIMD and GPGPU respectively than the baseline code.</p>
<br clear="left"/>

---

<h3>3) <a target="_blank" rel="noopener noreferrer" href="">Not a Drill</a></h3>
<img align="left" src="" width="330">
&nbsp;
<p align="justify">Make Scene::FindNearest and Scene::IsOccluded (defined in template/scene.h) as fast as possible, without changing the interface in any way. Anything is
permitted, as long as the '1. Basics' project and '2. Whitted' project can use
scene.h without changes. This probably excludes a GPGPU implementation. My implementation is around 35% faster than the baseline code.</p>
<br clear="left"/>
