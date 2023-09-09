import pstats

stats = pstats.Stats('profile_data.cprof')
stats.strip_dirs().sort_stats('time').print_stats(15)
