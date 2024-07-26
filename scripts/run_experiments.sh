echo ""
echo "Running wall-time experiment"
echo ""
time python experiments/measure_wall_time.py

echo ""
echo "Running memory experiment"
echo ""
time python experiments/measure_memory.py

echo ""
echo "Running stability experiment"
echo ""
time python experiments/measure_stability.py
