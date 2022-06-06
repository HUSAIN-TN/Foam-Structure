import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d

# Compute

def truncatedOctahedron(origin=[0,0,0],factor=1):
	vertex =  [[-1.5, -0.5, 0.],
			   [-1.5, 0.5, 0.],
			   [-1., -1., -0.707107],
			   [-1., -1., 0.707107],
			   [-1., 1., -0.707107],
			   [-1., 1., 0.707107],
			   [-0.5, -1.5, 0.],
			   [-0.5, -0.5, -1.41421],
			   [-0.5, -0.5, 1.41421],
			   [-0.5, 0.5, -1.41421],
			   [-0.5, 0.5, 1.41421],
			   [-0.5, 1.5, 0.],
			   [0.5, -1.5, 0.],
			   [0.5, -0.5, -1.41421],
			   [0.5, -0.5, 1.41421],
			   [0.5, 0.5, -1.41421],
			   [0.5, 0.5, 1.41421],
			   [0.5, 1.5, 0.],
			   [1., -1., -0.707107],
			   [1., -1., 0.707107],
			   [1., 1., -0.707107],
			   [1., 1., 0.707107],
			   [1.5, -0.5, 0.],
			   [1.5, 0.5, 0.]]

	faces = [[ 0, 1, 5,10, 8, 3],
			 [17,11, 4, 9,15,20],
			 [ 1, 0, 2, 7, 9, 4],
			 [14,16,21,23,22,19],
			 [20,15,13,18,22,23],
			 [ 3, 8,14,19,12, 6],
			 [ 2, 6,12,18,13, 7],
			 [ 1, 4,11, 5],
			 [ 8,10,16,14],
			 [ 0, 3, 6, 2],
			 [12,18,22,19],
			 [ 7,13,15, 9],
			 [23,21,17,20]]

	return (np.array(vertex))*factor+origin, faces

def pyritohedron(h,origin=[0,0,0],factor=1):
	"""
		Weare-Phelan h=0.6908
	"""
	h=h
	vB = 1-h**2
	vA = 1+h
	vertex =  list(it.product([1,-1],repeat=3))
	for ii in range(0,3):
		x = np.roll([[0],[vA,-vA],[vB,-vB]],ii)
		vertex += list(it.product(*x))

	faces = [[ 2,12, 0,16,17],
			 [17,16, 1,13, 3],
			 [ 7,11,10, 6,19],
			 [11, 3,17, 2,10],
			 [ 2,10, 6,14,12],
			 [ 0,12,14, 4, 8],
			 [ 1, 9, 5,15,13],
			 [ 3,11, 7,15,13],
			 [ 7,15, 5,18,19],
			 [18,19, 6,14, 4],
			 [ 8, 9, 5,18, 4],
			 [ 0,16, 1, 9, 8]]
	return np.array(vertex)*factor+origin,np.array(faces)

def tetradecahedron(origin=[0,0,0],factor=1):
	"""
		Vertex from: http://www.steelpillow.com/polyhedra/wp/wp.htm
	"""
	vertex=[[ 1.        ,  1.17480157,  1.58740237],
 			[-1.        ,  1.17480157,  1.58740237],
 			[-1.58740237,  0.        ,  1.58740237],
 			[-1.        , -1.17480157,  1.58740237],
 			[ 1.        , -1.17480157,  1.58740237],
 			[ 1.58740237,  0.        ,  1.58740237],
 			[ 1.33333545,  1.8414693 ,  0.25406692],
 			[-1.33333545,  1.8414693 ,  0.25406692],
 			[-2.17480475,  0.        ,  0.4126008 ],
 			[-1.33333545, -1.8414693 ,  0.25406692],
 			[ 1.33333545, -1.8414693 ,  0.25406692],
 			[ 2.17480475,  0.        ,  0.4126008 ],
 			[ 1.8414693 ,  1.33333545, -0.25406692],
 			[ 0.        ,  2.17480475, -0.4126008 ],
 			[-1.8414693 ,  1.33333545, -0.25406692],
 			[-1.8414693 , -1.33333545, -0.25406692],
 			[ 0.        , -2.17480475, -0.4126008 ],
 			[ 1.8414693 , -1.33333545, -0.25406692],
 			[ 1.17480157,  1.        , -1.58740237],
 			[ 0.        ,  1.58740237, -1.58740237],
 			[-1.17480157,  1.        , -1.58740237],
 			[-1.17480157, -1.        , -1.58740237],
 			[ 0.        , -1.58740237, -1.58740237],
 			[ 1.17480157, -1.        , -1.58740237]]

	faces =[[ 0, 1, 2, 3, 4, 5],
			[ 0, 5,11,12, 6],
			[ 0, 1, 7,13, 6],
			[ 1, 2, 8,14, 7],
			[ 2, 3, 9,15, 8],
			[ 3, 4,10,16, 9],
			[ 4, 5,11,17,10],
			[18,19,20,21,22,23],
			[23,18,12,11,17],
			[18,19,13, 6,12],
			[19,20,14, 7,13],
			[20,21,15, 8,14],
			[22,23,17,10,16],
			[21,22,16, 9,15]]
	return np.array(vertex)*factor+origin,np.array(faces)

def polyDraw(vertex,faces,axes,col=(0.2,0.7,0.2)):
	X = vertex[:,0]
	Y = vertex[:,1]
	Z = vertex[:,2]
	axes.scatter(X,Y,Z)
	"""
	for ii,point in enumerate(vertex):
		axes.text(point[0],point[1],point[2],ii)
	"""
	collection = Poly3DCollection([vertex[p] for p in faces])
	collection.set_facecolor(col)
	collection.set_alpha(0.8)
	axes.add_collection3d(collection)

def main():
	pyriv,pyrif = pyritohedron( 0.6908, [3,0,0] )
	tetrav,tetraf = tetradecahedron([-1,0,0],0.80 )
	truv,truf = truncatedOctahedron([-5,0,0])

	figure = plt.figure()

	axes = figure.add_subplot(111,projection='3d')
	plt.axis('off')
	axes.grid(None)
	axes.set_aspect('equal')
	polyDraw(pyriv,pyrif,axes)
	polyDraw(tetrav,tetraf,axes,(0.7,0.2,0.2))
	polyDraw(truv,truf,axes,(0.2,0.2,0.7))
	plt.show()	


if __name__ == '__main__':
	main()

