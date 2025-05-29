# hand-eye-calibration
 In this repository I will implement my own hand-eye calibration algorithm for both moving and static cameras


In fact you might also think 'Hannes, I don't own a robotic manipulator worth a few tenthousands of Euros!'. To you I say: 1) Poor you! and 2) Don't worry, we can use the same approaches I will be discussing here for other types of calibration, like finding an arbitrary pose of one camera with respect to another one. This is the setup for computing the position of an object using two cameras with disparity!

aruco motivation

## moving cam
```math
\mathbf{t}_\text{T} = \mathbf{t}_\text{TCP}(t) + \mathbf{R}_\text{TCP}(t) \mathbf{t}_\text{C} + \mathbf{R}_\text{TCP}(t) \mathbf{R}_\text{C} \mathbf{t}_\text{M}(t)
```
for the rotation the relationship is similar, however, we also have to consider the rotation of the target. On the TCP side we have the measurements of the target in the camera frame, given by rhe rotation $`\mathbf{R}_\text{M}`$ and rotation of the target in the base frame $`\mathbf{R}_\text{T}`$
```math
\mathbf{R}_\text{T} = \mathbf{R}_\text{TCP}(t) \mathbf{R}_\text{C} \mathbf{R}_\text{M}(t).
```
The reason we set up these relationship for the target is that the target is static, i.e., the value of the left hand side if the two equations is not changing over time. Consequently, the right hand side must be constant as well. This gives us a great starting point to finding the camera's position and rotation.
For a given guess of the camera rotation and translation in the TCP frame only the true answer will result in a constant corresponding target pose for all the different observed TCP poses. The issue is that we do not know the pose of the target. It's not like we couldn't, but we don't need to! And quiet frankly, I am lazy, and I don't want to. It is not easy to do and botching this will ruin the following camera calibration.
Instead, we will minimize the target's variance, possibly without even knowing the target's pose.
As we saw, we can decouple the translation from the orientation. And the expression of the translation depends on the initially unknown camera rotation $`\mathbf{R}_\text{C}`$. So we unfortunately have to start with the slightly more complicated rotations after tackling the translation $`\mathbf{t}_\text{C}`$ of the camera.

### rotation

The reason the rotation is slightly harder is that rotations are not only nonlinear but to parametrize them there are a few different ways, e.g., a rotation vector, quaternions or euler angles. I will use a rotation vector to pass to the optimizer later so I have a minimal representation of the Rotation without having to specify any equality constraints. However, this is only for implementation proposes, and all the maths will be done with the rotation matrices directly, so no need to do any maths with the rotation vectors directly. We will need to compute the variance of the expression $`\mathbf{R}_\text{T} = \mathbf{R}_\text{TCP}(t) \mathbf{R}_\text{C} \mathbf{R}_\text{M}(t)`$. There are may ways to do this, arguably the most simple is to convert the resulting matrix to rotation parameters and average over all the measurements we have. Then minimize the deviation of the rotation parameters to this obtained average rotation. The only flaw of this is that working with rotation parameters in this way only really works if the rotations are similar enough. A value of 30° is what i have seen floating around the internet the most. When we have a reasonable initial guess for the cameras orientation this is a fair assumption. But as I said I am lazy and i don't want to figure out how the camera might be oriented. Moreover, I want the algorithm to be robust enough to recover the real solution even for crazy initial guesses. Instead, we will be working with the matrices directly and compute the angle of the rotation $`\mathbf{I} = \mathbf{R}_\text{TCP}(t) \mathbf{R}_\text{C} \mathbf{R}_\text{M}(t) \mathbf{R}^\text{T}_\text{T}`$ which ideally should be close to zero. For the unknown Rotation $`\mathbf{R}_\text{T}`$ we will introduce an additional auxiliary optimization variable. To compute the angle of a rotation we will use the fact that
```math
\text{trace}(\mathbf{R}) = 1 + 2 \cos(\alpha),
```
so the sum of residuals we are trying to minimize is 
```math
J_\text{rot}(\mathbf{x})=\sum^{n}_{t=1} \text{acos}\left(\frac{1}{2}\left(\text{trace}(\mathbf{R}_\text{TCP}(t) \mathbf{R}_\text{C} \mathbf{R}_\text{M}(t) \mathbf{R}^\text{T}_\text{T})-1\right)\right)^2.
```
And with that we are ready to start optimizing. This will hopefully yield the rotation $`\mathbf{R}_\text{C}`$. This was quiet a lengthy process but I promise the following computations are not only easier but pretty much analogous, since we are still trying to minimize the quantities variances.

### translation
 be more clear what is done Linear least squares
As promised the approach for the translation $`\mathbf{t}_\text{C}`$ will be pretty similar, so I will try to keep things short. Starting with the initial expression
$`\mathbf{t}_\text{T} = \mathbf{t}_\text{TCP}(t) + \mathbf{R}_\text{TCP}(t) \mathbf{t}_\text{C} + \mathbf{R}_\text{TCP}(t) \mathbf{R}_\text{C} \mathbf{t}_\text{M}(t)`$ we want to minimize its variance. The rotation $`\mathbf{R}_\text{C}`$ now has been identified and only $`\mathbf{t}_\text{C}`$ and $`\mathbf{t}_\text{T}`$ remain unknown. Luckily translations are linear, so by defining $`\mathbf{A}_t := \mathbf{R}_\text{TCP}(t)`$ and $`\mathbf{b}_t := -\mathbf{t}_\text{TCP}(t) - \mathbf{R}_\text{TCP}(t) \mathbf{R}_\text{C} \mathbf{t}_\text{M}(t)`$ we get the residual
```math
\begin{bmatrix} \mathbf{A}_1 & -\bm{I} \\\ \vdots & \vdots \\\ \mathbf{A}_n & -\bm{I} \end{bmatrix} \begin{bmatrix} \mathbf{t}_\text{C} \\\ \mathbf{t}_\text{T} \end{bmatrix} - \begin{bmatrix} \mathbf{b}_1 \\\ \vdots \\\ \mathbf{b}_n \end{bmatrix}=:\mathbf{A}\mathbf{x}-\mathbf{b}
```
and as we want to minimize the variance, our residual becomes
```math
J_\text{trans}(\mathbf{x})=\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert_2^2.
```
So similar to the approach taken for the rotation an additional optimization variable $`\mathbf{t}_\text{T}`$ is introduced for the target. 
One can also skip the introduction of this auxiliary variable by minimizing the deviation from the parametrized mean directly. 
In my [previous post](https://github.com/heschmann/Multi-camera-disparity-in-computer-vision/tree/main) where I explored multi camera disparity in computer vision I used a similar approach. There I promised to show that they are actually the same thing under some assumptions. So if you are interested in learning more about this, hang tight and stay until the end!

## static cam

```math
\mathbf{R}_\text{TCP}(t) \mathbf{t}_\text{T} + \mathbf{t}_\text{TCP}(t) = \mathbf{t}_\text{C}(t) +  \mathbf{R}_\text{C} \mathbf{t}_\text{M}(t)
```

```math
\mathbf{R}_\text{TCP}(t) \mathbf{R}_\text{T} = \mathbf{R}_\text{C} \mathbf{R}_\text{M}(t).
```

### rotation
```math
\mathbf{R}_\text{T} = \mathbf{R}_\text{TCP}^\text{T}(t)\mathbf{R}_\text{C} \mathbf{R}_\text{M}(t).
```

so the sum of residuals we are trying to minimize is 
```math
J_\text{rot}(\mathbf{x})=\sum^{n}_{t=1} \text{acos}\left(\frac{1}{2}\left(\text{trace}(\mathbf{R}_\text{TCP}^\text{T}(t) \mathbf{R}_\text{C} \mathbf{R}_\text{M}(t) \mathbf{R}^\text{T}_\text{T})-1\right)\right)^2.
```

TCP is transposed now.

### translation
to have a similar structure as before we define $`\mathbf{A}_t := \mathbf{R}_\text{TCP}^\text{T}(t)`$ and $`\mathbf{b}_t := \mathbf{R}_\text{TCP}^\text{T}(t) \mathbf{t}_\text{TCP}(t) - \mathbf{R}_\text{TCP}^\text{T}(t) \mathbf{R}_\text{C}(t) \mathbf{R}_\text{C} \mathbf{t}_\text{M}(t)`$

## jointly
do both jointly (first rotation, add N residuals, then compute exact translation and feed in N times the residual, provide alpha for weighting both, maybe some error in rotation can lower the translation error)
The advantage of this is that we can include additional constraints. For example if we know that our camera is a certain distance from the TCP we can reject rotations that would not result in t_c of that magnitude. Can make it more robust
```math
J_\text{joint}(\mathbf{x}) = J_\text{rot}(\mathbf{x}) + \alpha J_\text{trans}(\mathbf{x})
```
Additionally, we can include additional information in the form of constraints! This might help guide the optimization towards the best solution and, e.g.,  penalizes camera and target positions that do not align with what we know about the robot-camera setup. A few things that you might be able to guess are the distance of the camera from the TCP/ the origin and the distance of the target from the origin/ the TCP. This corresponds to constraints of the form
```math
 d_\text{C} - \Delta d_{\text{C}}\leq\lVert\mathbf{t}_\text{C}\rVert_2\leq d_\text{C} + \Delta d_{\text{C}}
```
```math
 d_\text{T} - \Delta d_{\text{T}}\leq\lVert\mathbf{t}_\text{T}\rVert_2\leq d_\text{T} + \Delta d_{\text{T}}
```
where $`d_\text{C/T}\geq 0`$ is the guess for the distance of $`\mathbf{t}_\text{C/T}`$ and $`\Delta d_\text{C/T}\geq 0`$ is the uncertainty of the guess. So if we know the origin of the camera is 125mm $`\pm`$ 5mm from the TCP we would set $`d_\text{C}=0.125`$ and $`\Delta d_\text{C}=0.005`$. An even more insightful information would be a precise guess for the Cartesian position of these two quantities in their respective coordinate systems. These constraints take the form
```math
 \lVert\mathbf{t}_\text{C}-\mathbf{t}_\text{guess,C}\rVert_2\leq r_\text{C}
```
```math
 \lVert\mathbf{t}_\text{T}-\mathbf{t}_\text{guess,T}\rVert_2\leq r_\text{T}
```
where $`\mathbf{t}_\text{C/T,guess}`$ is the guess for the camera and target position and $`r_\text{C/T}\geq 0`$ is the radius of a ball around this guess we expect the true values to be in. Of course these four constraints can be combined. It is also possible to include other constraints, but these four are the ones I implemented. Note, that you should know what you are doing here, as specifying wrong prior information via these constraints can also drive away the optimizer form the true solution when the constraints are not reflected in reality. As a rule of thumb, I would suggest not specifying any constraints initially and then checking wether including them improves the solution (judging based in the residuals for translation and rotation). Moreover, the initial guess of the optimization (the rotation vectors of the camera and target) should ideally be chosen such that the constraints are initially satisfied.

When can it be beneficial to include constraints?
-Do you have high measurement noise (translation or rotation)
-Do you only have a low number of measurements?
-Do you have (approximate) additional information about the magnitude of the targets or cameras position vector or its rough position in Cartesian space?
-Does the sequential approach give bad results?
If any of these questions was answered with yes: Just give it a go!
Another factor is computational complexity. If you are impatient, have a high number of samples, or only need approximate results. Optimizing the rotation jointly with the translation is not something I would recommend.

## disparity by setting TCP = Marker1
for disparity: setup two cameras looking for the same Aruco marker
we do not have a TCP, in general.
The measurement vector and rotation is taken from the Acuco measurement of the second camera.
The TCP vector and orientation is the Aruco measurement of the first camera. The Aruco marker effectively becomes our TCP and all the methods can be applied as usual. So instead of the forward kinematics of the robotic manipulator, we have the Aruco measurement of the first camera.
Other similar applications are virtual reality, where you overlay images of an object onto the camera frame by identifying the cameras pose as described above and the objects position using simulation data for example. Here we need to project the simulated object onto the camera frame, which not only requires the cameras pose in Cartesian space (called its extrinsics), but also the cameras intrinsics.

## Appendix

### We Need to Move the TCP! (And not move the Target)
Suppose we look at the camera fixed to the TCP again, but instead of moving the TCP, we move the target, at the end of the day moving a marker is simpler than moving a robotic arm and logging its TCP coordinates, right?
We obtain the now slightly modified matrix equation 
```math
\mathbf{I} = \mathbf{R}_\text{TCP} \mathbf{R}_\text{C} \mathbf{R}_\text{M}(t) \mathbf{R}^\text{T}_\text{T}(t).
```
Let us look at two different time steps now 
```math
\mathbf{R}_\text{TCP} \mathbf{R}_\text{C} \mathbf{R}_\text{M}(t_1) \mathbf{R}^\text{T}_\text{T}(t_1) = \mathbf{I} = \mathbf{R}_\text{TCP} \mathbf{R}_\text{C} \mathbf{R}_\text{M}(t_2) \mathbf{R}^\text{T}_\text{T}(t_2).
```
what now?
if fact if we would not move the tcp in our other two approaches we would not 
be able to differentiate cam and TCP! So varying the TCP orientation is important! This and specifying a good initial guess for our TCP rotation is our main way for dealing with measurement noise! To make the calibration even more robust, we could include even more prior information to guide the optimization, like an initial guess for the vector t_C or its magnitude.

### Equivalence of Some Translation Residuals
```math
\mathbf{t}_\text{T} = \mathbf{A}_t \mathbf{t}_\text{C} - \mathbf{b}_t,
```
or in matrix form
```math
\begin{bmatrix} \mathbf{A}_1 \\\ \vdots \\\ \mathbf{A}_n \end{bmatrix} \mathbf{t}_\text{C} - \begin{bmatrix} \mathbf{b}_1 \\\ \vdots \\\ \mathbf{b}_n \end{bmatrix}=:\mathbf{A}\mathbf{t}_\text{C} - \mathbf{b}.
```
This should work! But there is another way to write the same thing operating on our original expression directly, using the so called centering matrix
```math
\mathbf{H}:= \begin{bmatrix} \frac{n-1}{n}\mathbf{I} & \dots &  \frac{-1}{n}\mathbf{I}\\\ \vdots & \ddots & \vdots \\\ \frac{-1}{n}\mathbf{I} & \dots &  \frac{n-1}{n}\mathbf{I} \end{bmatrix} = \mathbf{I}_{3n} - \frac{1}{n} \bm{1}\bm{1}^\text{T} \otimes \mathbf{I}_{3}
```
with Krockecker product 
left with centering matrix multiplication effectively centers or data in a block-wise fashion
```math
\mathbf{H}(\mathbf{A}\mathbf{t}_\text{C} - \mathbf{b})
```
 by shifting everything by the mean over all the time steps
```math
\mathbf{t}_\text{T} \approx \frac{1}{n}\sum^{n}_{t=1} \mathbf{A}_t \mathbf{t}_\text{C} - \frac{1}{n}\sum^{n}_{t=1}\mathbf{b}_t =: \tilde{\mathbf{A}}_t \mathbf{t}_\text{C} - \tilde{\mathbf{b}}_t.
```
One question one might ask is wether this approach gives a similar result vs. the approach taken before. The answer is YES! In fact the two (or three approaches depending on wether you count the explicit subtraction of the mean separately) approaches are equivalent. In the following, I want to motivate why.
Remember that
```math
\begin{bmatrix} \mathbf{A}_1 & -\bm{I} \\\ \vdots & \vdots \\\ \mathbf{A}_n & -\bm{I} \end{bmatrix} \begin{bmatrix} \mathbf{t}_\text{C} \\\ \mathbf{t}_\text{T} \end{bmatrix} - \begin{bmatrix} \mathbf{b}_1 \\\ \vdots \\\ \mathbf{b}_n \end{bmatrix}=:\mathbf{A}\mathbf{x}-\mathbf{b}
```
and
```math
J_\text{trans}(\mathbf{x})=\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert_2^2 = \sum_{t=1}^{n}\lVert\mathbf{A}_i\mathbf{t}_\text{C}-\mathbf{b}_i-\mathbf{t}_\text{T}\rVert_2^2.
```
We want that the partial derivative wrt. our optimization variable is zero. So let us take the partial derivative with respect to the target $`\mathbf{t}_\text{T}`$
```math
\frac{\partial J_\text{trans}}{\partial\mathbf{t}_\text{T}}=0=-2\sum_{t=1}^{n}(\mathbf{A}_i\mathbf{t}_\text{C}-\mathbf{b}_i-\mathbf{t}_\text{T}) = n\mathbf{t}_\text{T} -\sum_{t=1}^{n}(\mathbf{A}_i\mathbf{t}_\text{C}-\mathbf{b}_i)
```
and rearranging for the target $`\mathbf{t}_\text{T}`$
```math
\mathbf{t}_\text{T}=\frac{1}{n}\sum_{t=1}^{n}\mathbf{A}_i\mathbf{t}_\text{C}-\frac{1}{n}\sum_{t=1}^{n}\mathbf{b}_i = \tilde{\mathbf{A}}_t \mathbf{t}_\text{C} - \tilde{\mathbf{b}}_t
```
we obtain the mean parametrized by the camera position $`\mathbf{t}_\text{C}`$. This is what we substituted in the alternative approach and what we implicitly did when using the centering matrix.