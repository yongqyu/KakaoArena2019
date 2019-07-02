import numpy as np
import torch
from tqdm import tqdm

from models import GMF

dev_users_path = '/data/private/Arena/datasets/predict/dev.users'
test_users_path = '/data/private/Arena/datasets/predict/test.users'
id2reader = np.load('/data/private/Arena/prepro_results/id2reader.npy')
reader2id = np.load('/data/private/Arena/prepro_results/reader2id.npy', allow_pickle=True).item()
item_list = np.load('/data/private/Arena/prepro_results/item_list.npy')
writerid2items = np.load('/data/private/Arena/prepro_results/writerid2items.npy', allow_pickle=True).item()
freq_item = ['@brunch_141', '@brunch_151', '@brunch_145', '@tenbody_1305', '@intlovesong_28', '@dailylife_207', '@hyehyodam_19', '@steven_179', '@brunch_140', '@sangheeshyn_66', '@brunch_142', '@deckey1985_51', '@conbus_43', '@sweetannie_145', '@dailylife_219', '@tenbody_1164', '@seochogirl_1', '@brunch_144', '@x-xv_19', '@honeytip_945', '@brunch_147', '@brunch_105', '@conbus_35', '@mightysense_8', '@honeytip_940', '@tenbody_1297', '@shanghaiesther_46', '@brunch_143', '@studiocroissant_43', '@chofang1_15', '@anti-essay_133', '@brunch_149', '@brunch_148', '@brunch_133', '@nareun_143', '@seochogirl_6', '@peregrino97_779', '@brunch_2', '@bzup_281', '@brunch_1', '@noey_130', '@zorbayoun_14', '@kam_60', '@anti-essay_124', '@mightysense_9', '@seochogirl_16', '@kam_33', '@dailylife_173', '@seochogirl_18', '@minimalmind88_26', '@wootaiyoung_85', '@jmg5308_163', '@seochogirl_7', '@seochogirl_8', '@seochogirl_17', '@seochogirl_3', '@wikitree_54', '@psychiatricnews_18', '@seochogirl_2', '@kecologist_68', '@nareun_134', '@brunch_111', '@nplusu_49', '@dancingsnail_21', '@dong02_1372', '@seochogirl_5', '@roysday_279', '@seochogirl_10', '@zheedong_3', '@mongul-mongul_76', '@brunch_152', '@jooyoon_51', '@seochogirl_9', '@taekangk_44', '@tenbody_1034', '@greenut90_85', '@boot0715_36', '@seochogirl_11', '@hjl0520_26', '@boot0715_39', '@kangsunseng_754', '@pyk8627_41', '@seochogirl_12', '@seochogirl_4', '@pureleyy_8', '@merryseo_73', '@dryjshin_245', '@bong_362', '@tenbody_1306', '@yeoneo_125', '@seochogirl_13', '@roysday_307', '@merryseo_53', '@maama_170', '@seochogirl_14', '@shoong810_216', '@cathongzo_90', '@tenbody_902', '@seouledu_84']

hidden_dim = 128
batch_size = 128
num_keywords = 96892; num_readers = 310758; num_writers = 19065
model = GMF(num_readers, num_writers, num_keywords, hidden_dim, writerid2items).cuda()
model.load_state_dict(torch.load('./models/8_gmf.pkl'))
model.eval()

file_w = open('./recommend.txt', 'w')
file = open(dev_users_path, 'r')
data_ = file.readlines()
lines = []
unk_line_indices = {}
for i, line in enumerate(tqdm(data_)):
    if len(lines) < batch_size:
        if reader2id.get(line.strip()) != None:
            lines.append(reader2id[line.strip()])
        else:
            unk_line_indices[i] = line.strip()
    else:
        tensor_lines = torch.from_numpy(np.array(lines)).cuda()
        preds = model.predict(tensor_lines).tolist()
        i = 0
        for (line, pred) in zip(lines, preds):
            while i in unk_line_indices.keys():
                file_w.write(unk_line_indices[i]+' '+' '.join(freq_item)+'\n')
                i += 1
            file_w.write(id2reader[line]+' '+' '.join(list(map(lambda x: item_list[x], pred)))+'\n')
            i += 1

        liens = []

tensor_lines = torch.from_numpy(np.array(lines)).cuda()
preds = model.predict(tensor_lines).tolist()
i = 0
for (line, pred) in zip(lines, preds):
    while i in unk_line_indices.keys():
        file_w.write(unk_line_indices[i]+' '+' '.join(freq_item)+'\n')
        i += 1
    file_w.write(id2reader[line]+' '+' '.join(list(map(lambda x: item_list[x], pred)))+'\n')
    i += 1

file_w.close()
file.close()
