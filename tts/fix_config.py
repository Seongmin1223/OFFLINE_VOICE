import json 
path=r'C:\MeloTTS-Windows\melo\data\example\config.json' 
c=json.load(open(path)) 
c['train']['batch_size']=4 
c['train']['segment_size']=8192 
open(path,'w').write(json.dumps(c,indent=2)) 
