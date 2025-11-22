# audioShieldNet/asnet_5/audioshieldnet/data/__init__.py



from .asvspoof import build_dataloaders as _build_asvspoof
from .asvspoof21_split import build_dataloaders as _build_asv21_split
from .librisevoc import build_dataloaders as _build_lsv
from .codecfake import build_dataloaders as _build_codecfake
from .fakeOrReal import build_dataloaders as _build_for
from .wavefake import build_dataloaders as _build_wavefake
from .codecfake_split import build_dataloaders as _build_codecfake_split
from .librisevoc_split import build_dataloaders as _build_lsv_split
from .multi import build_dataloaders as _build_multi 





def build_dataloaders(cfg):
    name = str(cfg.get("data", {}).get("name", "asvspoof21")).lower()
    if name in ("asvspoof21", "asv21", "asvspoof"):
        return _build_asvspoof(cfg)
    
    if name in ("asvspoof21_split","asv21_split","asv21_80_10_10"): 
        return _build_asv21_split(cfg)
    
    if name in ("librisevoc", "lsv", "librisecvoc", "libri_se_voc"):
        return _build_lsv(cfg)
    if name in ("codecfake", "codec-fake", "codec_fake"):
        return _build_codecfake(cfg)
    if name in ("for", "forgerynet-for"):  
        return _build_for(cfg)   
    if name in ("wavefake", "wave-fake", "wave_fake"):         
        return _build_wavefake(cfg)
    
    if name in ("codecfake", "codecfake_split", "codec-fake"):
        return _build_codecfake_split(cfg)
    if name in ("librisevoc_split", "lsv_split", "libri_split"):  
        return _build_lsv_split(cfg)
    
    if name in ("multi","mix","cross","multi_cross"): 
        return _build_multi(cfg)

    raise ValueError(f"Unknown dataset name: {name}. Supported: asvspoof21, librisevoc, codecfake, for, wavefake.")
