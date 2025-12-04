# python test.py --bin ./cpu
# python test.py --bin "python cpu.py"
# or customize your testing command

python test.py --bin "python y86_simulator_opt.py"

#此外，若要通过 python y86_simulator_opt.py < test/prog1.yo > answer/prog1.json 实现读写文件，请在cmd中操作