# dlanomaly
Deep Learning methods for Anomaly Detection

El proyecto contiene las siguientes etapas:
1.	Ajuste y otptimización de redes profundas
2.	Comprobación y comparación de los modelod usandos un conjunto de test.

## Results
| | | | | | | | | | | | |
|-|-|-|-|-|-|-|-|-|-|-|-|
| | | | | | | | | | | | |
| |**Model architecture|Supervision|Objective|Activation|Input|Hidden Neurons /layer|Loss|Class weights|AUC|Recall @100|Recall @500**|
| |Isolation Forest (baseline) {'max_samples': 131072}|Unsupervised|Anomaly score| |29| | | |0.954|43%|72%|
| |VAE|Unsupervised|Reconstruction|Tanh|29|20,10,5,10,20|MSE| |0.963|26%|53%|
| |Conditional VAE|Unsupervised|Reconstruction|lReLU|29|20,10,5,10,20|MSE| |0.938|28%|55%|
| | | | | | | | | | | | |
| |LR (baseline) {'newton-cg', 'l2', 'C':100}|Spervised|Classification| |29| |BCE| |0.976|80%|90%|
| |DNN|Spervised|Classification|ReLU|29|64, 64|BCE| |0.948|76%|90%|
| |DNN|Spervised|Classification|ReLU|29|64, 64|BCE|Balanced|0.983|72%|90%|
| |DNN Keras tuner|Spervised|Classification|ReLU|29|64, 96|BCE| |0.953|79%|90%|
| |DNN Autokeras|Spervised|Classification|ReLU|29|512, 1024|BCE| |0.933|**84%**|90%|
| | | | | | | | | | | | |
| |Conditional VAE latent|Semi Superised|Reconstruction + Classification|lReLU|5|20,10,5 => 64,64| |Balanced|0.978|66%|87%|
| |Conditional VAE + DNN merged|Semi Superised|Reconstruction + Classification|lReLU|[29, 5]|[64, 10], [1]| |Balanced|**0.99**|77%|89%|
