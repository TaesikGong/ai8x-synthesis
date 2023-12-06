python ai8xize.py --verbose --test-dir exp --prefix cifar100-simplewide2x-q --device MAX78000 --checkpoint-file trained/ai85-cifar100-simplenetwide2x-qat-mixed-q.pth.tar --config-file networks/cifar100-simplewide2x.yaml --display-checkpoint --compact-data --overwrite --debug-latency --timer 0

python ai8xize.py --verbose --test-dir exp --prefix cifar100-simplewide2x-q_0_1 --device MAX78000 --checkpoint-file trained/ai85-cifar100-simplenetwide2x-qat-mixed-q.pth.tar --config-file networks/cifar100-simplewide2x.yaml --display-checkpoint --compact-data --overwrite --debug-latency --timer 0 --stop-after 2

python ai8xize.py --verbose --test-dir exp --prefix cifar100-simplewide2x-q_0_5 --device MAX78000 --checkpoint-file trained/ai85-cifar100-simplenetwide2x-qat-mixed-q.pth.tar --config-file networks/cifar100-simplewide2x.yaml --display-checkpoint --compact-data --overwrite --debug-latency --timer 0 --stop-after 6
