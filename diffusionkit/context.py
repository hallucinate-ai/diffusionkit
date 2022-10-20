progress_callback = None
progress_stages = [
	('init', 1),
	('encode', 2),
	('sample', 1),
	('decode', 1)
]

class DiffusionContext:
	def __init__(self, params, image):
		self.params = params
		self.image = image
		self.sampling_steps = 0
		self.progress_callback = progress_callback
		self.progress_stage = None
		self.progress_sample = 0
		self.finished = False


	def report_stage(self, stage):
		self.progress_stage = stage
		self.dispatch_progress()


	def report_sampling_steps(self, steps):
		self.sampling_steps = steps


	def make_sampling_iter(self, iter):
		if not self.progress_callback:
			return iter

		for item in iter:
			self.progress_stage = 'sample'
			self.progress_sample += 1
			self.dispatch_progress()
			yield item


	def dispatch_progress(self):
		if not self.progress_callback:
			return

		total_points = sum([weight for _, weight in progress_stages])
		total_points += self.sampling_steps

		stage_index = [stage for stage, _ in progress_stages].index(self.progress_stage)
		stage_index = max(0, stage_index - 1)

		if self.finished:
			points = total_points
		else:
			points = sum([weight for _, weight in progress_stages[0:stage_index]])
			points += self.progress_sample

		self.progress_callback(points / total_points, self.progress_stage)


	def finish(self):
		self.finished = True
		self.dispatch_progress()
		pass



class progress_tracking():
	def __init__(self, callback):
		self.callback = callback


	def __enter__(self):
		global progress_callback
		progress_callback = self.callback
		

	def __exit__(self, type, value, traceback):
		global progress_callback
		progress_callback = None