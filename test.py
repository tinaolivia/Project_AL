import torch
import csv
import train


def test(train_set, val_iter, model ,args):
    
    with open('accuracies/{}_{}.csv'.format(args.method, args.dataset), mode='w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['Train Size', 'Accuracy'])
    
    for al_iter in range(args.rounds):
        
        print('\nLoading model {}, method {}, train size {} ... \n'.format(al_iter, args.method,
                                                                           len(train_set)+al_iter*args.inc))
        model.load_state_dict(torch.load('{}/al_{}_{}.pt'.format(args.method, args.dataset, al_iter)))
        accuracy = train.evaluate(val_iter, model, args)
        
        with open('accuracies/{}_{}.csv'.format(args.method,args.dataset), mode='a') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([len(train_set) + al_iter*args.inc, accuracy])

