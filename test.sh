# python test.py --bin ./cpu
# python test.py --bin "python cpu.py"
# or customize your testing command

# 不启用 cache,x=1/2/3/4/5/6/7/8/9/10
#执行以下指令后output_opt.json中即为test/progx.yo的模拟结果
python y86_simulator_opt.py test/progx.yo > output_opt.json
python compare_opt_json.py progx

# 启用 cache,x=1/2/3/4/5/6/7/8/9/10
python y86_simulator_opt.py test/progx.yo --cache > output_opt.json
python compare_opt_json.py progx