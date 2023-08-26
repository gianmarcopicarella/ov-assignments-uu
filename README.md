# Optimization & Vectorization 22/23 course assignments

You can find a **assignment.pdf** document detailing my algorithms and the corresponding implementations in each assignment's folder.

<h3>1) <a target="_blank" rel="noopener noreferrer" href="https://github.com/gianmarcopicarella/ov-assignments-uu/tree/main/draw_the_line/optimized">Draw the Line</a></h3>
<img align="left" src="https://raw.githubusercontent.com/gianmarcopicarella/ov-assignments-uu/main/readme/draw_the_line.jpg?token=GHSAT0AAAAAACGZC3RUQRGOV2TACZNVJGWCZHKIHCQ" width="330">
&nbsp;
<p align="justify">For the purpose of this assignment, you are asked to optimize line rendering alone; 
you can thus leave all other code unmodified. Please do not use the GPU to speed up line rendering, 
and do not use multi-threading.</p>
<br clear="left"/>

---

<h3>2) <a target="_blank" rel="noopener noreferrer" href="https://github.com/gianmarcopicarella/ov-assignments-uu/tree/main/velocity_vector/optimized">Velocity Vector (~2.1x & 466x faster with SIMD and GPGPU)</a></h3>
<img align="left" src="https://raw.githubusercontent.com/gianmarcopicarella/ov-assignments-uu/main/readme/velocity_vector.jpg?token=GHSAT0AAAAAACGZC3RV4UYPY2PAYNOTQTXSZHKIHXQ" width="330">
&nbsp;
<p align="justify"> The code for this assignment implements a basic cloth simulation on a 256x256 grid using a technique known as Verlet integration.
The goal of this assignment is to improve the performance of the application using vectorization. This can be accomplished with two main technologies discussed during the course: SIMD and GPGPU.</p>
<br clear="left"/>

---

<h3>3) <a target="_blank" rel="noopener noreferrer" href="https://github.com/gianmarcopicarella/ov-assignments-uu/tree/main/not_a_drill/optimized">Not a Drill (~35% faster)</a></h3>
<img align="left" src="https://raw.githubusercontent.com/gianmarcopicarella/ov-assignments-uu/main/readme/not_a_drill.jpg?token=GHSAT0AAAAAACGZC3RVK3UCTTPZF7BKXKQWZHKIIAQ" width="330">
&nbsp;
<p align="justify">Make Scene::FindNearest and Scene::IsOccluded (defined in template/scene.h) as fast as possible, without changing the interface in any way. Anything is
permitted, as long as the '1. Basics' project and '2. Whitted' project can use
scene.h without changes. This probably excludes a GPGPU implementation.</p>
<br clear="left"/>
