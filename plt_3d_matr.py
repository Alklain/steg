# importing matplotlib.pyplot from
# python
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pl
# importing numpy package from python
import numpy as np

# creating an empty figure for plotting
fig = plt.figure()

# defining a sub-plot with 1x2 axis and defining
# it as first plot with projection as 3D
ax = fig.add_subplot(1, 1, 1, projection='3d')

# creating a range of values for
# x1,y1  from -1 to 1 with
# a space of 0.1 between the elements so that
# we can create a single curve in the plot
x1 = np.arange(-1, 1, 0.1)
y1 = np.arange(-1, 1, 0.1)

# Creating a mesh grid with x ,y and x1,
# y1 which creates an n-dimensional
# array
x1, y1 = np.meshgrid(x1, y1)

# Creating a cosine function with the
# range of values from the meshgrid
z1 = np.cos(x1 * np.pi / 2)

# Creating a wireframe plot with the points
# x1,y1,z1 along with the plot line as red
#ax.plot_surface(x1, y1, z1, color="red")
ax.plot_surface(z1, color="red")
# Creating my second subplot with 1x2 axis and defining
# it as the second plot with projection as 3D
#ax = fig.add_subplot(1, 2, 2, projection='3d')

# defining a set of points for X1,Y1 and Z1
#X1 = [1, 2, 1, 4, 3, 2, 7, 5, 9]
#Y1 = [8, 2, 7, 4, 3, 6, 1, 8, 9]
#Z1 = [1, 2, 4, 7, 9, 6, 7, 6, 9]

# Plotting 3 points X1,Y1,Z1 with
# color as purple
#ax.plot_trisurf(X1, Y1, Z1, color='purple')

# Showing the above plot
plt.show()