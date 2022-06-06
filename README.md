# Simultaneous Contact Location and Object Pose Estimation Using Proprioceptive Tactile Feedback
**[Project Website](https://www.mmintlab.com/research/scope)**

**[Project Video](https://youtu.be/rAfFP-LJ7So)**

Imagine that a robot picks up two objects to complete a task such as assembly or insertion. 
It will almost always pick them up with some sort of pose uncertainty. 
In this paper, we address the challenge of grasped object localization using only the sense of touch. 
With our method, the robot can bring two objects into contact and, from feel, guess their in-hand poses. 
To accomplish this, we propose a novel state-estimation algorithm that jointly estimates contact location and 
object pose in 3D using exclusively proprioceptive tactile feedback. 
Our approach leverages two complementary particle filters: one to estimate contact location (CPFGrasp) and another to estimate object poses (SCOPE).
We implement and evaluate our approach on real-world single-arm and dual-arm robotic systems.
We demonstrate how by bringing two objects into contact, the robots can infer contact location and object poses simultaneously.

![Project Image](https://user-images.githubusercontent.com/60672716/171698259-389e7f3f-99ba-4602-80b6-633250e41b31.png)
