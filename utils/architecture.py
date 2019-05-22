#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

# This has been designed according to: https://arxiv.org/pdf/1409.1556.pdf
class SimpleVGG16(nn.Module):
    def __init__(self, params, num_classes=None):
        super(SimpleVGG16, self).__init__()
        # Layer 1
        self.i2c1 = nn.Conv2d(in_channels=3,
                              out_channels=16, # number of (ouput) feature maps                            
                              kernel_size=3,                        
                              padding=1,                         
                              stride=1,
                              bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.p1 = nn.MaxPool2d(kernel_size=2,
                               padding=0,
                               stride=2, # defaults to stride=kernel_size
                               return_indices=False)
    
        # Layer 2
        self.c2 = nn.Conv2d(in_channels=16,
                            out_channels=32, # number of (ouput) feature maps                            
                            kernel_size=3,                        
                            padding=1,                      
                            stride=1,
                            bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.p2 = nn.MaxPool2d(kernel_size=2,
                               padding=0,
                               stride=2, # defaults to stride=kernel_size
                               return_indices=False)
        
        # Layer 3
        self.c3_1 = nn.Conv2d(in_channels=32,
                            out_channels=64, # number of (ouput) feature maps                            
                            kernel_size=3,                        
                            padding=1,                      
                            stride=1,
                            bias=True)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.c3_2 = nn.Conv2d(in_channels=64,
                            out_channels=64, # number of (ouput) feature maps                            
                            kernel_size=3,                        
                            padding=1,                      
                            stride=1,
                            bias=True)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.p3 = nn.MaxPool2d(kernel_size=2,
                               padding=0,
                               stride=2, # defaults to stride=kernel_size
                               return_indices=False)
        
        
        # Layer 4
        self.c4_1 = nn.Conv2d(in_channels=64,
                            out_channels=128, # number of (ouput) feature maps                            
                            kernel_size=3,                        
                            padding=1,                      
                            stride=1,
                            bias=True)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.c4_2 = nn.Conv2d(in_channels=128,
                            out_channels=128, # number of (ouput) feature maps                            
                            kernel_size=3,                        
                            padding=1,                      
                            stride=1,
                            bias=True)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.p4 = nn.MaxPool2d(kernel_size=2,
                               padding=0,
                               stride=2, # defaults to stride=kernel_size
                               return_indices=False)
        
        
        # Layer 5
        self.c5_1 = nn.Conv2d(in_channels=128,
                            out_channels=128, # number of (ouput) feature maps                            
                            kernel_size=3,                        
                            padding=1,                      
                            stride=1,
                            bias=True) 
        self.bn5_1 = nn.BatchNorm2d(128)
        self.c5_2 = nn.Conv2d(in_channels=128,
                            out_channels=128, # number of (ouput) feature maps                            
                            kernel_size=3,                        
                            padding=1,                      
                            stride=1,
                            bias=True)
        self.bn5_2 = nn.BatchNorm2d(128)
        self.p5 = nn.MaxPool2d(kernel_size=2,
                               padding=0,
                               stride=2, # defaults to stride=kernel_size
                               return_indices=False)
        
       
        if num_classes is not None:
            self.h1 = nn.Linear(5 * 5 * 128, 512)
            self.h12bn = nn.BatchNorm1d(512)
            self.drop1 = nn.Dropout(params['dropout'])
            self.h2 = nn.Linear(512, 512)
            self.h22bn = nn.BatchNorm1d(512)
            self.drop2 = nn.Dropout(params['dropout'])
            self.h2y = nn.Linear(512, num_classes)
        else:
            self.h1 = nn.Linear(5 * 5 * 128, 512)
            self.h12bn = nn.BatchNorm1d(512)
            self.drop1 = nn.Dropout(params['dropout'])
            self.h2 = nn.Linear(512, 512)
            self.h22bn = nn.BatchNorm1d(512)
            self.drop2 = nn.Dropout(params['dropout'])
            self.h2y = None

        
        
        
    def forward(self, input, ):
        # Layer 1
        p1 = self.p1(nn.functional.relu(self.bn1(self.i2c1(input))))
        # Layer 2
        p2 = self.p2(nn.functional.relu(self.bn2(self.c2(p1))))
        # Layer 3
        p3 = self.p3(
            nn.functional.relu(
            self.bn3_2(self.c3_2(
                nn.functional.relu(
                    self.bn3_1(self.c3_1(p2)))))))
        # Layer 4
        p4 = self.p4(
            nn.functional.relu(
                self.bn4_2(self.c4_2(
                    nn.functional.relu(
                        self.bn4_1(self.c4_1(p3)))))))
        # Layer 5
        p5 = self.p5(
            nn.functional.relu(
                self.bn5_2(self.c5_2(
                    nn.functional.relu(
                        self.bn5_1(self.c5_1(p4)))))))
        
        
        p5 = p5.view(-1, 5 * 5 * 128)
        
        h1 = self.drop1(nn.functional.relu(self.h12bn(
            self.h1(p5))))
        h2 = self.drop2(nn.functional.relu(self.h22bn(
            self.h2(h1))))
        if self.h2y is not None:
            h2y = self.h2y(h2)
            pred = nn.functional.log_softmax(h2y, dim=1)
        
        return pred if self.h2y is not None else h2


