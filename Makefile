format-and-lint:
	pre-commit run --all-files

test:
	pytest

clean-preview:
	git clean -xdn

clean:
	git clean -xdf

run-experiments:
	time python experiments/measure_wall_time.py
	time python experiments/measure_memory.py
	time python experiments/measure_stability.py
	time python experiments/estimate_parameters.py
