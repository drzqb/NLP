'''
    pkuseg加载自定义字典进行分词
'''
import pkuseg

seg=pkuseg.pkuseg()
print(' '.join(seg.cut('恶性肿瘤的分期越高，患者预后越差。通过对肿瘤不同恶性程度的划分，TNM分期在预测预后方面更加有效。')))

seg=pkuseg.pkuseg(user_dict='dict.txt')
print(' '.join(seg.cut('恶性肿瘤的分期越高，患者预后越差。通过对肿瘤不同恶性程度的划分，TNM分期在预测预后方面更加有效。')))
