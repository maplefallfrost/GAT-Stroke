import numpy as np
import torch as th
import time
from sklearn.metrics import confusion_matrix
from dataset import to_device

def print_result(count):
    num_classes = count.shape[0]
    if num_classes == 5:
        id_to_tag = ['Graph', 'Text', 'Table', 'List', 'Math']
    elif num_classes == 2:
        id_to_tag = ['Non-text', 'Text']
    
    # Confusion matrix with accuracy for each tag
    print (("{: >2}{: >9}{: >9}%s{: >9}" % ("{: >9}" * num_classes)).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(num_classes)] + ["Percent"]))
    )
    for i in range(num_classes):
        print (("{: >2}{: >9}{: >9}%s{: >9}" % ("{: >9}" * num_classes)).format(
            str(i), id_to_tag[i], str(count[i].sum()),
            *([count[i][j] for j in range(num_classes)] +
              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        ))

    # Global accuracy
    accuracy = 100. * count.trace() / max(1, count.sum())
    print ("Stroke accuracy: %i/%i (%.5f%%)" % (
        count.trace(), count.sum(), accuracy)
    )

    
def evaluate(model, device, loader, num_classes, name):
    model.eval()
    print(name + ":")
    count = np.zeros((num_classes, num_classes), dtype=np.int32)
    duration = 0
    for it, batch in enumerate(loader):
        fg, lg = to_device(batch, device)
        start = time.time()
        with th.no_grad():
            logits = model(fg)
        _, predictions = th.max(logits, dim=1)
        duration += time.time() - start
        labels = lg.ndata['y']
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        count += confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    
    model.train()
    
    print("Time cost: {:.4f}s".format(duration))
    print_result(count)
    
    accuracy = 100. * count.trace() / max(1, count.sum())
    return accuracy, count
