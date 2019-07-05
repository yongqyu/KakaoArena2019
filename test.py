import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from models import GMF, RNN

dev_users_path = '/data/private/Arena/datasets/predict/dev.users'
test_users_path = '/data/private/Arena/datasets/predict/test.users'
#rnn_test_data = np.load('/data/private/Arena/prepro_results/rnn_test_data.npy', allow_pickle=True).item()
rnn_test_data = np.load('/data/private/Arena/prepro_results/rnn_test_data.npy')
dev_mask = torch.from_numpy(np.load('/data/private/Arena/prepro_results/dev_mask.npy')).float().cuda()
id2reader = np.load('/data/private/Arena/prepro_results/id2reader.npy')
reader2id = np.load('/data/private/Arena/prepro_results/reader2id.npy', allow_pickle=True).item()
item_list = np.load('/data/private/Arena/prepro_results/item_list.npy')
valid_tensor = torch.load('/data/private/Arena/prepro_results/valid_writer_keywd.pkl').cuda()
writerid2items = np.load('/data/private/Arena/prepro_results/writerid2items.npy', allow_pickle=True).item()
freq_item = ['@brunch_141', '@brunch_151', '@brunch_145', '@tenbody_1305', '@intlovesong_28', '@dailylife_207', '@hyehyodam_19', '@steven_179', '@brunch_140', '@sangheeshyn_66', '@brunch_142', '@deckey1985_51', '@conbus_43', '@sweetannie_145', '@dailylife_219', '@tenbody_1164', '@seochogirl_1', '@brunch_144', '@x-xv_19', '@honeytip_945', '@brunch_147', '@brunch_105', '@conbus_35', '@mightysense_8', '@honeytip_940', '@tenbody_1297', '@shanghaiesther_46', '@brunch_143', '@studiocroissant_43', '@chofang1_15', '@anti-essay_133', '@brunch_149', '@brunch_148', '@brunch_133', '@nareun_143', '@seochogirl_6', '@peregrino97_779', '@brunch_2', '@bzup_281', '@brunch_1', '@noey_130', '@zorbayoun_14', '@kam_60', '@anti-essay_124', '@mightysense_9', '@seochogirl_16', '@kam_33', '@dailylife_173', '@seochogirl_18', '@minimalmind88_26', '@wootaiyoung_85', '@jmg5308_163', '@seochogirl_7', '@seochogirl_8', '@seochogirl_17', '@seochogirl_3', '@wikitree_54', '@psychiatricnews_18', '@seochogirl_2', '@kecologist_68', '@nareun_134', '@brunch_111', '@nplusu_49', '@dancingsnail_21', '@dong02_1372', '@seochogirl_5', '@roysday_279', '@seochogirl_10', '@zheedong_3', '@mongul-mongul_76', '@brunch_152', '@jooyoon_51', '@seochogirl_9', '@taekangk_44', '@tenbody_1034', '@greenut90_85', '@boot0715_36', '@seochogirl_11', '@hjl0520_26', '@boot0715_39', '@kangsunseng_754', '@pyk8627_41', '@seochogirl_12', '@seochogirl_4', '@pureleyy_8', '@merryseo_73', '@dryjshin_245', '@bong_362', '@tenbody_1306', '@yeoneo_125', '@seochogirl_13', '@roysday_307', '@merryseo_53', '@maama_170', '@seochogirl_14', '@shoong810_216', '@cathongzo_90', '@tenbody_902', '@seouledu_84', '@kotatsudiary_66']
#freq_item = ['@brunch_151' '@sweetannie_145' '@chofang1_15' '@seochogirl_1' '@seochogirl_16' '@seochogirl_18' '@seochogirl_17' '@conbus_43' '@tenbody_1305' '@brunch_152' '@seochogirl_11' '@hjl0520_26' '@seochogirl_12' '@intlovesong_28' '@seochogirl_13' '@seochogirl_14' '@dailylife_207' '@seochogirl_15' '@wootaiyoung_85' '@seochogirl_10' '@steven_179' '@seochogirl_28' '@seochogirl_20' '@seochogirl_29' '@noey_130' '@shindong_38' '@seochogirl_8' '@shanghaiesther_46' '@tenbody_1164' '@seochogirl_7' '@seochogirl_6' '@mothertive_66' '@seochogirl_2' '@seochogirl_9' '@seochogirl_3' '@deckey1985_51' '@kotatsudiary_66' '@bzup_281' '@seochogirl_4' '@roysday_314' '@hongmilmil_33' '@seochogirl_5' '@ohmygod_42' '@boot0715_115' '@hyehyodam_19' '@hjl0520_28' '@wikitree_54' '@fuggyee_108' '@brunch_149' '@syshine7_57' '@mightysense_9' '@roysday_313' '@sweetannie_146' '@onyouhe_98' '@roysday_307' '@ladybob_30' '@13july_92' '@dryjshin_255' '@aemae-human_15' '@dailylife_219' '@tamarorim_133' '@sunnysohn_60' '@keeuyo_57' '@anetmom_52' '@ladybob_29' '@moment-yet_155' '@yoriyuri_12' '@dong02_1372' '@kidjaydiary_6' '@curahee_7' '@thinkaboutlove_234' '@thebluenile86_4' '@scienceoflove_5' '@hjl0520_27' '@jijuyeo_9' '@anti-essay_150' '@13july_94' '@seochogirl_41' '@dryjshin_256' '@aemae-human_9' '@mentorgrace_8' '@anetmom_47' '@psychiatricnews_18' '@namgizaa_46' '@dailylife_178' '@boot0715_111' '@moment-yet_157' '@keeuyo_56' '@kam_65' '@honeytip_945' '@choyoungduke_157' '@jinbread_111' '@dreamwork9_25' '@kam_60' '@dancingsnail_65' '@kyungajgba_60' '@syshine7_56' '@dancingsnail_64' '@anti-essay_153' '@mariandbook_413']
hidden_dim = 256
batch_size = 128
num_keywords = 96894; num_readers = 310759; num_writers = 19066; num_items = 643105
model = RNN(num_readers, num_writers, num_keywords, num_items,
            hidden_dim, valid_tensor, writerid2items).cuda()
model.load_state_dict(torch.load('./models/2_rnn_gmf.pkl'))
model.eval()

file_w = open('./recommend.txt', 'w')
file = open(dev_users_path, 'r')
readers = file.read().splitlines()
rnn_test_data = data.TensorDataset(torch.from_numpy(rnn_test_data))
dev_loader = data.DataLoader(rnn_test_data, batch_size=batch_size, shuffle=False)
# [item_id, writer] + keywd + [reg_ts, meg_id]
with torch.no_grad():
    for i, input in enumerate(tqdm(dev_loader)):
        preds = model(input[0].cuda())
        preds = torch.mul(preds, dev_mask[i*batch_size:(i+1)*batch_size])

        split_size = 1024
        hidden_dot_product = [torch.sort(preds[:,split_size*k:split_size*(k+1)], descending=True)[1][:,:100] + k*split_size
                              for k in range((preds.size(1)//split_size)+1)]
        hidden_dot_product = torch.cat(hidden_dot_product, 1)
        sorted_dot_product = torch.sort(preds.gather(1, hidden_dot_product), descending=True)[1][:,:100]
        sorted_dot_product = hidden_dot_product.gather(1, sorted_dot_product)

        for reader, pred in zip(readers[i*batch_size:],sorted_dot_product):
            file_w.write(reader+' '+' '.join(list(map(lambda x: item_list[x], pred)))+'\n')

file_w.close()
file.close()
