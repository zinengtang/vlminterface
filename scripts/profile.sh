# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" \
# python3 profile_performance.py --suite scales_vlm_toggle --cadence 32

# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" \
# python3 profile_performance.py --suite multi_agent

# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" \
# python3 profile_performance.py --suite planning_modes_all_scales

xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" \
python3 profile_performance.py --suite vlm_token_numbers_all_scales


# xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@" \
# python3 profile_performance.py --suite all



# # 1) VLM vs no-VLM across scales (cadence 32), bound 2/7/72
# python profile_controller_and_vlm_extended.py --suite scales_vlm_toggle --cadence 32

# # 2) VLM vs no-VLM for cadences 8/32/128 at medium
# python profile_controller_and_vlm_extended.py --suite cadence_medium --cadences 8 32 128

# # 3) Online vs Offline planning (medium, cadence 32)
# python profile_controller_and_vlm_extended.py --suite planning_modes_medium --cadence 32

# # 4) Multi-agent (2 agents) across scales with three regimes
# python profile_controller_and_vlm_extended.py --suite multi_agent --num-agents 2 --cadence 32

# # Run everything (default)
# python profile_controller_and_vlm_extended.py --suite all
