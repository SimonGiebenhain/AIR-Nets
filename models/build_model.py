#from . import Onet
#from . import IFnet
#from . import ConvONet
#from . import AIRnet
#
#
#method_dict = {
#    'onet': Onet,
#    'ifnet': IFnet,
#    'convonet': ConvONet,
#    'airnet': AIRnet,
#    'pointnet++': AIRnet
#}
#
#def build_model(CFG):
#    enc_method = CFG['encoder']['type']
#    encoder = method_dict[enc_method].get_encoder(CFG)
#    dec_method = CFG['decoder']['type']
#    decoder = method_dict[dec_method].get_decoder(CFG)
#    return encoder, decoder

def build_model(CFG):
    enc_method = CFG['encoder']['type']
    if enc_method in ['airnet', 'pointnet++']:
        from . import AIRnet
        encoder = AIRnet.get_encoder(CFG)
    elif enc_method == 'ifnet':
        from . import IFnet
        encoder = IFnet.get_encoder(CFG)
    elif enc_method == 'convonet':
        from . import ConvONet
        encoder = ConvONet.get_encoder(CFG)
    elif enc_method == 'onet':
        from . import Onet
        encoder = Onet.get_encoder(CFG)
    else:
        raise ValueError('Unknown encoder type: ' + enc_method + '!!')

    dec_method = CFG['decoder']['type']
    if dec_method in ['airnet', 'pointnet++']:
        from . import AIRnet
        decoder = AIRnet.get_decoder(CFG)
    elif dec_method == 'ifnet':
        from . import IFnet
        decoder = IFnet.get_decoder(CFG)
    elif dec_method == 'convonet':
        from . import ConvONet
        decoder = ConvONet.get_decoder(CFG)
    elif dec_method == 'onet':
        from . import Onet
        decoder = Onet.get_decoder(CFG)
    else:
        raise ValueError('Unknown decoder type: ' + dec_method + '!!')
    return encoder, decoder
