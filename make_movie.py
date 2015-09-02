import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

def make_movie(a_evolution, xaxis):
	FFMpegWriter = manimation.writers['ffmpeg']
	metadata = dict(title='Movie Test', artist='Matplotlib',
	        comment='Movie support!')
	writer = FFMpegWriter(fps=15, metadata=metadata)

	fig = plt.figure()
	l, = plt.plot([], [], 'k-o')

	print len(a_evolution[0])
	print len(xaxis)
	with writer.saving(fig, "writer_test.mp4", 100):
	    for step in a_evolution:
	    	plt.plot(xaxis, step[1:-1])
	    	writer.grab_frame()
	    	#plt.show()
	    	plt.clf()