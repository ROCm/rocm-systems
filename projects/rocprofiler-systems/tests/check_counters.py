import sys
from perfetto.trace_processor import TraceProcessor

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <perfetto-trace.proto>")
    sys.exit(1)

tp = TraceProcessor(trace=sys.argv[1])

print("=== All counter tracks ===")
for row in tp.query("SELECT id, name FROM counter_track"):
    print(f"  {row.id}: {row.name}")

print("\n=== Counter totals ===")
for name in ["VCN Activity", "JPEG Activity", "VCN Busy", "JPEG Busy"]:
    for row in tp.query(
        f"""SELECT SUM(counter.value) AS total_value
            FROM counter_track JOIN counter ON counter.track_id = counter_track.id
            WHERE counter_track.name LIKE '%{name}%'"""
    ):
        val = row.total_value if row.total_value is not None else 0
        print(f"  {name}: {val}")
