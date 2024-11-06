from omegaconf import OmegaConf

from deploy import get_parser
from main import instantiate_from_config
from generator.src.modules.losses.metrics import *

###


if __name__ == "__main__":

    # Load configuration
    option, unknown = get_parser().parse_known_args()
    config = OmegaConf.merge(*[OmegaConf.load(cfg) for cfg in [option.config]], OmegaConf.from_dotlist(unknown))
    
    # Instantiate and load data
    x = instantiate_from_config(config.data);  x.prepare_data();  x.setup()
    
    xrec_config = config.data

    # RGB-RGB :: RAW-RAW
    # xrec_config.params.train.params.training_images_list_file = xrec_config.params.train.params.training_images_list_file.split('.')[0] + '_rec.txt'
    # xrec_config.params.validation.params.validation_images_list_file = xrec_config.params.validation.params.validation_images_list_file.split('.')[0] + '_rec.txt'
    # xrec_config.params.test.params.test_images_list_file = xrec_config.params.test.params.test_images_list_file.split('.')[0] + '_rec.txt'

    # RGB-RAW :: RAW-RGB (Including dataloader changes over label path)
    xrec_config.params.train.params.training_images_list_file = xrec_config.params.train.params.training_images_list_file.split('.')[0].replace('rgb-rgb', 'rgb-raw') + '_rec.txt'
    xrec_config.params.validation.params.validation_images_list_file = xrec_config.params.validation.params.validation_images_list_file.split('.')[0].replace('rgb-rgb', 'rgb-raw') + '_rec.txt'
    xrec_config.params.test.params.test_images_list_file = xrec_config.params.test.params.test_images_list_file.split('.')[0].replace('rgb-rgb', 'rgb-raw') + '_rec.txt'

    xrec = instantiate_from_config(xrec_config);  xrec.prepare_data();  xrec.setup()

    for x_loader, xrec_loader, dataset_type in zip([x.train_dataloader(), x.val_dataloader(), x.test_dataloader()],  # 
                                                   [xrec.train_dataloader(), xrec.val_dataloader(), xrec.test_dataloader()], # 
                                                   ['Train', 'Validation', 'Test']):

        # Frechet Inception Distance
        fid = frechet_inception_distance(x_loader, xrec_loader)

        ssim = 0; psnr = 0; l2 = 0
        for _, (x_elem, xrec_elem) in tqdm(enumerate(zip(x_loader, xrec_loader))):  # Expected batch size 1

            x_elem = x_elem['image'][0]
            xrec_elem = xrec_elem['image'][0]

            # Reconstruction Loss        
            l2 += torch.nn.functional.mse_loss(x_elem, xrec_elem)

            x_elem = (x_elem * 255).type(torch.uint8).numpy()
            xrec_elem = (xrec_elem * 255).type(torch.uint8).numpy()

            # Structural Similarity
            ssim += structural_similarity(x_elem, xrec_elem)  

            # Peak Signal to Noise ratio    
            psnr += peak_signal_noise_ratio(x_elem, xrec_elem)

        ssim /= len(x_loader); psnr /= len(x_loader); l2 /= len(x_loader)

        print('{0} Set\tFID: {1}\tL2: {2}\tSSIM: {3}\tPSNR: {4}'.format(dataset_type, fid, l2, ssim, psnr))
