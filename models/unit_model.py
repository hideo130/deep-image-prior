class UnitModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.loss_names = []

        self.loss_total = opt.gan_w * self.loss_adv_a + \
                          opt.gan_w * self.loss_adv_b  

    
    def forward(self):
        pass


    def optimize_parameters(self):
        self.forward()