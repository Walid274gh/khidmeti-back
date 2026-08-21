#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# ml/generate_dataset.py — P1/P2 : dataset synthétique darija pour le classifieur
# 2 têtes (intent + profession) — voir plan khid-ai-darija-pipeline.
#
#   python3 ml/generate_dataset.py
#
# Écrit :
#   ml/dataset/labels.json      liste gelée des labels (P0, +5 métiers P2b) —
#                               source de vérité pour Kaggle (P2) et ai-nlu (P3)
#   ml/dataset/synth_v5.csv     text,intent,profession,source — 100% train
#
# v2 (2026-08-03) : le run Kaggle v1 a donné 64.6% profession sur eval_heldout
# (gap de domaine : gabarits vs problème-d'abord). Remède = lexique problèmes
# ×3 (pannes + INSTALLATIONS + confusables : clim voiture→mechanic, machine qui
# fuit→appliance), équilibre ar/lat 50/50, négatifs durs OOS (réparation
# téléphone, dentiste, taxi, achat, formation), app_question élargi (paiement,
# offline, profil). eval_heldout.csv reste intouchable (assert 0 overlap).
#
# v5 (2026-08-04) : test terrain — «daw magtou3» → plumber (faux). Cause racine :
# les gabarits latins n'utilisaient QUE les emprunts français (courant, fuite…),
# jamais les mots darja romanisés (daw, magtou3, y9atter). Remède : axe de
# TRANSLITTÉRATION — 35% des lignes latines sont une ligne arabe entière
# translittérée en arabizi (dico mots fréquents + repli lettre-à-lettre avec
# brisure des grappes de consonnes). Les deux lexiques deviennent liés.
#
# v3 = P2b (2026-08-03) :
#   - 5 nouveaux métiers : plasterer (BA13), welder (soudeur), barber (coiffeur),
#     tailor (couturier), caterer (traiteur). Le coiffeur SORT de l'OOS (il était
#     un négatif dur en v2 — désormais in-scope) ; apprendre la soudure/couture
#     reste OOS (demande de formation, pas de service).
#   - Registres régionaux : centre (base), ouest/Oran (ghaya, nta3, nebghi),
#     Tlemcen (qaf→hamza : قهوة→اهوة, arabizi 9→2), est/Constantine (yasser,
#     kifah, dorka), sud (drouk, zin). Swaps lexicaux post-remplissage.
#   - Axe orthographique du qaf : ar ق→{ق,ڨ,گ}, arabizi 9→{9,q,g} (+2 à Tlemcen).
#   - Interjections : ياخو/ya kho et sœurs dans préfixes/suffixes.
#
# ponytail: générateur par gabarits — plafond connu = diversité syntaxique
# limitée; upgrade = passe de paraphrase gros-modèle sur Kaggle si P2 < 85%.
# ══════════════════════════════════════════════════════════════════════════════
import csv
import json
import random
from collections import Counter
from pathlib import Path

random.seed(20260803)

OUT = Path(__file__).resolve().parent / 'dataset'

INTENTS = ['find_worker', 'urgent_service', 'price_inquiry',
           'app_question', 'greeting_chitchat', 'out_of_scope']

# Doit rester identique à VALID_PROFESSIONS (intent-extractor) + seeder.
# v3.1 (P2b-bis) : gardener RETIRÉ du catalogue — le jardinage devient un
# négatif dur OOS (même mécanique que coiffeur avant P2b, en sens inverse).
PROFESSIONS = ['none', 'plumber', 'electrician', 'ac_repair', 'mason',
               'painter', 'carpenter', 'cleaner', 'appliance_repair',
               'mover', 'mechanic',
               'plasterer', 'welder', 'barber', 'tailor', 'caterer']

# ── Lexique professions : noms + problèmes (pannes ET installations), ar/lat ──
LEX = {
    'plumber': {
        'ar_names': ['سباك', 'بلومبيي', 'مول السباكة'],
        'lat_names': ['plombier', 'plombi', 'sebbak'],
        'ar_problems': ['طوفان الما في الكوزينة', 'الروبيني يقطر', 'البالوعة مسدودة',
                        'الشوفو ما يسخنش الما', 'فويت دو تحت اللافابو', 'الدوش مقطوع الما',
                        'قنوات الما تاع الدار قدام يسقطو', 'اللافابو راه يفويتي',
                        'الما حابس في الدار كامل', 'الطواليت محبوسة والما يطلع',
                        'السيفون تاع التواليت خسر', 'الما ضعيف ما يطلعش للطابق',
                        'نحب نركب لافابو جديد في الحمام', 'نحب نبدل الروبينات تاع الكوزينة',
                        'كاين ريحة مجاري في الحمام', 'الما يقطر من سقف الحمام',
                        'نحتاج نركب سخان الما جديد', 'البانيو مسدود والما ما ينزلش'],
        'lat_problems': ["fuite d'eau f la cuisine", 'robinet y9atter', 'canalisation msdouda',
                         'chauffe-eau ma yeskhonch', 'fuite ta7t le lavabo', 'douche ma fihach el ma',
                         'lavabo rah mfwiti', 'roubini rah yfuiti', 'el ma 7abes f dar kamla',
                         'toilette bouchée w el ma ytla3', "la chasse d'eau khasra",
                         'pression ta3 el ma dai3fa bezzaf', 'nheb nrakeb lavabo jdid fel hammam',
                         'nbeddel les robinets ta3 cuisine', 'ri7at el mjari fel hammam',
                         'installation chauffe-eau jdid', 'baignoire msdouda',
                         'compteur ta3 el ma yfuiti'],
    },
    'electrician': {
        'ar_names': ['كهربائي', 'تريسيان', 'مول الضو'],
        'lat_names': ['électricien', 'electricien', 'trisyan', 'kahrabji'],
        'ar_problems': ['الضو مقطوع في الدار كامل', 'الديجونكتور يطيح ديما', 'بريز محروقة',
                        'الضو يشعل ويطفي وحدو', 'التابلو الكهربائي قديم ويخوف', 'ريحة كابل محروق',
                        'نحب نزيد بريزة جديدة في البيرو', 'نحب نركب تريات في السقف',
                        'نحتاج نبدل الكابلاج القديم تاع الدار', 'كي نشعل الماشينة يطيح الضو',
                        'نحب نركب لامبات ليد في الدار كامل', 'الانتيريتور خسر ما يشعلش الضو',
                        'نحتاج تركيب سونيري تاع الباب', 'الكونتور يدق كي نشعلو زوج ماشينات'],
        'lat_problems': ['toute la maison sans courant', 'disjoncteur yti7 kol chwiya',
                         'prise ma7rou9a', 'lampe techmel w tetfa wa7edha', 'tableau electrique 9dim',
                         'nzid des prises fel cuisine', 'installer des spots fel plafond',
                         'nrakeb un lustre f salon', 'changer le cablage 9dim',
                         'interrupteur khasser', 'installation sonnette ta3 el bab',
                         'compteur yti7 ki nch3el la clim w le four ensemble',
                         'les lampes yrigliw wa7edhom'],
    },
    'ac_repair': {
        'ar_names': ['مول الكليمة', 'تقني كليماتيزور', 'لي يصلح الكليمة'],
        'lat_names': ['technicien clim', 'reparateur climatiseur', 'mol clima'],
        'ar_problems': ['الكليمة ما تبردش', 'الكليمة تقطر الما في البيت', 'الكليماتيزور يدير صوت غريب',
                        'نحتاج تركيب كليمة جديدة', 'الكليمة تطلق ريحة كريهة',
                        'الكليمة تجري الما من الداخل', 'نحب نبدل بلاصة الكليمة',
                        'الكليمة ناقصها غاز', 'الكليمة تجمد وتحبس وحدها',
                        'نحتاج نتوايا وصيانة الكليمة قبل الصيف', 'الكليمة ما تسخنش في الشتا'],
        'lat_problems': ['clim ma tberredch', 'clima t9atter el ma', 'clim ydir bruit bizarre',
                         'installation clim jdida', 'clim tetlok ri7a khayba',
                         'la clim na9esha gaz', 'recharge gaz ta3 la clim',
                         'nettoyage w entretien ta3 la clim', 'la clim tjamed w t7bes wa7edha',
                         'déplacer la clim l chambre okhra', 'la clim ma tsakhanch f chta'],
    },
    'mason': {
        'ar_names': ['بنّاء', 'ماصون', 'مول البناء'],
        'lat_names': ['maçon', 'mason', 'bennay'],
        'ar_problems': ['نحب نبني حيط في الجنان', 'الحيط فيه شقوق كبار', 'نحتاج نزيد بيت فوق السطح',
                        'نصب دالة جديدة', 'الرطوبة كلات الحيط تاع البيت',
                        'نحب نديار كارلاج جديد في الكوزينة', 'نهدمو حيط ونحلو باب',
                        'السطح يقطر ونحتاج ايتونشيتي', 'نحب نديار سياج حول الجنان',
                        # v4 : ciblage faiblesse eval (mason 6/11) — pannes béton/carrelage/marches
                        'الكارلاج تاع الحمام تقلع وولى يجمع الما', 'حيط الحوش مايل وخايفين يطيح',
                        'نحب نعلي حيط الحوش متر زيادة', 'البلاط تاع الساحة تحرك وتكسر',
                        'نديرو مارشات قدام باب الدار', 'نبنيو غرفة زيادة للضياف في الحوش',
                        'سقف الغاراج طايح منو الاسمنت وباين الحديد'],
        'lat_problems': ['construire un mur fel jardin', 'fissures kbar fel 7it',
                         'dalle jdida', 'extension ta3 dar', 'humidité fel mur',
                         'refaire le carrelage ta3 cuisine', 'casser un mur w n7elou bab',
                         'etancheite ta3 stah', 'cloture ta3 jardin', 'crepissage ta3 façade',
                         'le carrelage ta3 hammam t9alla3', 'mur ta3 7ouch mayel',
                         'ndiro des marches 9oddam el bab', 'construire un garage f jardin',
                         'refaire la dalle ta3 la terrasse', 'monter un mur en parpaing',
                         'la chape ta3 les chambres 9bel le parquet'],
    },
    'painter': {
        'ar_names': ['صباغ', 'دهان', 'مول الصباغة'],
        'lat_names': ['peintre', 'sebbagh'],
        'ar_problems': ['نحب نصبغ الدار قبل العيد', 'الصباغة تقشرت في البيت', 'نحتاج صباغة السقف',
                        'صباغة الأبواب والفنيترات', 'نحب نبدل لون الصالون',
                        'كاين رطوبة وتقشير في سقف البيت', 'نحب صباغة غرفة الأطفال بألوان شابة',
                        'الحيوط مخبشين ونحتاجو تصليح وصباغة'],
        'lat_problems': ['peinture ta3 dar kamla', 'la peinture t9echret', 'peindre le plafond',
                         'sbagha ta3 les portes', 'nbeddel couleur ta3 salon',
                         "peinture ta3 chambre d'enfants", 'les murs mkharbchin ye7tajo enduit',
                         'repeindre la façade ta3 dar'],
    },
    'carpenter': {
        'ar_names': ['نجار', 'منوزيي'],
        'lat_names': ['menuisier', 'najjar'],
        'ar_problems': ['الباب تاع الكوزينة تكسر', 'نحب نديار پلاكار على قياس', 'الفنيترة ما تسكرش مليح',
                        'الطابلة تاع الصالون محتاجة تصليح', 'نحتاج أرفف للبيرو',
                        'نحب نديار كوزينة خشب على قياس', 'الپاركي تاع البيت تخرب',
                        'الدرج الخشبي يزقزق', 'نحب نركب باب رئيسية جديدة'],
        'lat_problems': ['porte mkessra', 'placard sur mesure', 'fenetre ma tetsakkarch mli7',
                         'meuble ye7taj tsli7', 'des etageres l biro',
                         'cuisine en bois sur mesure', 'parquet mkhareb',
                         'escalier en bois yzegzeg', 'reparer les volets ta3 fenetres'],
    },
    'cleaner': {
        'ar_names': ['عاملة تنظيف', 'فام دو ميناج', 'لي يدير الميناج'],
        'lat_names': ['femme de ménage', 'agent de nettoyage', 'menage'],
        'ar_problems': ['نحتاج تنظيف الدار قبل ما نسكنو فيها', 'تنظيف بعد خدمة البناء',
                        'نفض الغبرة ومسح البلاط تاع الفيلا',
                        'تنظيف الفور والتلاجة والكوزينة الكل قبل رمضان',
                        'الكناپي ولا مصفر من الاولاد ويحتاج غسيل عميق',
                        'ميناج أسبوعي للفيلا', 'تنظيف الموكيت والكناپي', 'الدار محتاجة تنظيف عميق',
                        'نحتاج تنظيف الزجاج والفنيترات تاع الفيلا', 'تنظيف الدار بعد العرس',
                        'نحتاج مرا تعاون ماما في قضيان الدار'],
        'lat_problems': ['nettoyage ta3 dar kamla', 'menage apres travaux', 'menage hebdomadaire',
                         'nettoyer canapé w moquette', 'nettoyage profond',
                         'nettoyage des vitres ta3 villa', 'menage ba3d la fete',
                         'nettoyage ta3 les bureaux', 'femme de menage teji kol jem3a',
                         # v4 : ciblage faiblesse eval (cleaner 7/10) — incl. contexte piège
                         # "déménagement" dans une demande de MÉNAGE
                         'nettoyage appartement 9bel ma nekriwah', 'grand menage ta3 printemps',
                         'nettoyage ta3 dar ba3d le demenagement',
                         'femme de menage l villa kol khemis'],
    },
    'appliance_repair': {
        'ar_names': ['مصلح الأجهزة', 'لي يصلح الماشينات', 'مصلح تلاجات'],
        'lat_names': ['réparateur électroménager', 'reparateur machine a laver'],
        'ar_problems': ['الماشينة تاع الصابون ما تدورش', 'التلاجة ما تبردش والمأكلة تخسر',
                        'الفور ما يسخنش', 'اللافيسال تحبس في نص البروغرام', 'الميكروند مات',
                        'الماشينة تاع الصابون تجري الما من تحت', 'التلاجة تدير صوت غريب',
                        'الكوزينيار ما يشعلوش البلاكات تاعو', 'الكونجيلاتور ما يجمدش'],
        'lat_problems': ['machine a laver ma tkhdemch', 'frigo ma ybaredch', 'four ma yeskhonch',
                         'lave-vaisselle y7bes f nos', 'micro-onde mat',
                         'machine a laver tejri el ma men ta7t', 'frigo ydir bruit bizarre',
                         'cuisiniere ma tech3elch', 'congelateur ma yjamedch',
                         'seche-linge khasser'],
    },
    'mover': {
        'ar_names': ['ناقل العفش', 'لي ينقل الأثاث', 'ديمناجور'],
        'lat_names': ['déménageur', 'demenageur', 'transport meubles'],
        'ar_problems': ['نحب ننقل العفش لدار جديدة', 'نقل تلاجة وماشينة للطابق الثالث',
                        'نبدلو الدار الشهر الجاي ونحتاجو نقل القش',
                        'نقل الحوايج من الدار القديمة للجديدة',
                        'نحتاج عمال ينزلولي العفش من الطابق الخامس',
                        'ديمناجمون كامل للدار', 'نقل مكتب من وسط المدينة', 'نقل عفش عروسة',
                        'نحتاج كاميو صغير ننقل بيه الأثاث', 'نقل بيانو من الطابق الرابع',
                        'نحول من وهران للعاصمة ونحتاج نقل العفش'],
        'lat_problems': ['déménagement complet', 'n9el les meubles l dar jdida',
                         'transporter frigo w machine', 'demenagement ta3 bureau',
                         'camion sghir bach nne9lou les meubles', 'transport ta3 piano',
                         'demenagement men Oran l Alger', 'n9el machine a laver l etage',
                         # v4 : ciblage faiblesse eval (mover 7/10)
                         'transport ta3 les affaires l dar jdida', 'location camion m3a demenageurs',
                         'des ouvriers ynazlou el 9ach mel 5eme etage',
                         'n7awlou l dar jdida f nefs el 7ay'],
    },
    'mechanic': {
        'ar_names': ['ميكانيسيان', 'ميكانيكي', 'مول الميكانيك'],
        'lat_names': ['mécanicien', 'mecanicien', 'mikanisyan'],
        'ar_problems': ['الطوموبيل ما تحبش تشعل', 'الفرينات يديرو صوت يخوف', 'الموتور يسخن بزاف',
                        'نحتاج فيدونج وتبديل الفيلترات', 'لاباتري راهي ميتة', 'العجلة تنفّس',
                        'كليمة الطوموبيل ما تبردش', 'الطوموبيل تدخن دخان كحل',
                        'الاومبرياج يزلق', 'الطوموبيل تحبس في نص الطريق',
                        'ضو تاع الموتور شاعل في الطابلو'],
        'lat_problems': ['voiture ma techa3lch', 'les freins ydiro bruit', 'moteur yeskhon bezzaf',
                         'vidange w les filtres', 'batterie morte', 'la roue tneffes',
                         'clim ta3 tonobil ma tberredch', 'la voiture tdakhen dokhan k7el',
                         'embrayage yzle9', 'voyant moteur allumé fel tableau',
                         'la voiture t7bes f nos tri9'],
    },
    'plasterer': {
        'ar_names': ['مول البلاكو', 'بلاكيست', 'مول الجبس'],
        'lat_names': ['plaquiste', 'mol l placo', 'plakist'],
        'ar_problems': ['نحب نديار فو بلافون بلاكو في الصالون', 'نديرو حيط بلاكو نقسمو بيه البيت',
                        'السقف تاع البلاكو طاح منو طرف', 'نحب ديكور جبس في السقف',
                        'تركيب بلاكو بلاتر مع السبوتات', 'نحب نخبي الكابلاج بحيط بلاكو',
                        'فو بلافون مع اضاءة مخبية', 'الجبس تاع السقف تشقق ومحتاج تصليح',
                        'نحب نديار مكتبة في الحيط بالبلاكو',
                        'جبس السقف طايح شوية شوية', 'حفرة كبيرة في حيط الجبس',
                        'نغطيو البيام تاع البلافون بالجبس', 'الرطوبة قوست البلاكو تاع البيت',
                        'الجبس تاع البيت طاح مع الشتا'],
        'lat_problems': ['faux plafond ba13 f salon', 'cloison placo bach n9assmou la chambre',
                         'plafond placo tah mennou tarf', 'decor jebs fel plafond',
                         'placo avec spots led', 'cloison ba13 avec porte',
                         'faux plafond avec lumiere indirecte', 'le platre ta3 plafond tcheqqeq',
                         'meuble tv en placo fel mur', 'isolation avec ba13',
                         'le placo tqawes mel humidité', 'trou kbir fel mur en platre',
                         'jebs ta3 plafond tah chwiya', 'cache poutre en ba13'],
    },
    'welder': {
        'ar_names': ['سودور', 'لحّام', 'مول السودير'],
        'lat_names': ['soudeur', 'soudor', '7addad'],
        'ar_problems': ['نحب نديار باب حديد للحوش', 'السياج تاع الحديد تكسر ومحتاج سودير',
                        'نحب حماية حديد للفنيترات', 'الباب الحديدية خرجت من بلاصتها',
                        'نديرو درابزين حديد للدرج', 'نحب بورطاي حديد للغاراج',
                        'تصليح سودير في السلم الحديدي', 'نحب نديار قفص حديد فوق السطح',
                        'البورطاي ما عادش يتسكر مليح'],
        'lat_problems': ['porte en fer forgé lel 7ouch', 'portail ta3 garage mkasser',
                         'grille de protection lel fenetres', 'rampe d escalier en fer',
                         'cloture en fer te7taj soudure', 'balustrade lel balcon',
                         'le portail khrej men blasstou', 'soudure ta3 porte metallique',
                         'nheb ndir porte blindée', 'reparation grille ta3 fenetre'],
    },
    'barber': {
        'ar_names': ['حلاق', 'كوافور', 'مول الحلاقة'],
        'lat_names': ['coiffeur', 'barbier', 'kwafur'],
        'ar_problems': ['نحب نحفف قبل العيد', 'نحتاج كوافور يجي للدار نهار العرس',
                        'تحفيفة وتدريج شاب للولد', 'حلاقة اللحية والتحديد',
                        'شعري طويل ونحب تحفيفة ديغراديه', 'نحتاج حلاق للعريس صباح العرس',
                        'تحفيفة للأولاد قبل الدخول المدرسي', 'نحب صبغة ومشطة في الدار',
                        'حلاقة في الدار لشيخ كبير ما يقدرش يخرج',
                        'حففلي راسي ديغراديه', 'الطفل يحتاج يحفف قبل المدرسة',
                        'نحفف وندير اللحية عند الدار', 'راسي شعث بزاف ونحتاج نحفف'],
        'lat_problems': ['coupe degradé m3a trait', 'coiffure a domicile pour le mariage',
                         'coupe + barbe w contour', 'ta7fifa lel 3id',
                         'coiffeur yji l dar', 'degradé lel weld',
                         'coupe pour les enfants avant la rentrée', 'brushing w couleur f dar',
                         'coiffure ta3 la mariée a domicile',
                         'n7affef 3and dar', 'ta7fifa w l7ya', 'drari y7afffo 9bel l3id'],
    },
    'tailor': {
        'ar_names': ['خياط', 'خياطة', 'مول الخياطة'],
        'lat_names': ['couturier', 'couturiere', 'khayat'],
        'ar_problems': ['نحب نخيط قندورة للعيد', 'السروال طويل ونحب نقصرو',
                        'ترقيع السروال لي تقطع من الركبة', 'رتوش خفيف على روبة السهرة',
                        'نحب نخيط ريدوات جداد للبيت', 'الكم تاع الفيستة طويل ويحتاج تقصير',
                        'تضييق القميجة من الجناب باش تجي على القد',
                        'السحاب تاع الجاكيطة خسر', 'نحب نخيط كسوة تقليدية للعرس',
                        'نحب نضيق الروبة من الجناب', 'خياطة ريدو للصالون',
                        'نحب نفصل كوستيم على قياسي', 'الكبوط تقطع من الكتف ومحتاج رتوش',
                        'نحب نخيط جبة قسنطينية للعرس',
                        'السروال يحتاج تقصير', 'الروبة تحتاج تضييق من الجناب',
                        'الكشابية محتاجة ترقيع', 'ترقيع الجاكيطة القديمة',
                        'تقصير الكم تاع القميجة'],
        'lat_problems': ['retouche pantalon twil', 'nkhayet 9andoura lel 3id',
                         'fermeture eclair ta3 veste khasra', 'robe soirée sur mesure',
                         'retouche ta3 costume', 'rideaux lel salon sur mesure',
                         'caftan lel 3ers', 'serwal twil nheb n9assrou',
                         'nheb nfassel karakou lel 3ers',
                         'ta9sir ta3 serwal', 'tarqi3 ta3 kachabiya',
                         'retouche w tadyiq ta3 roba',
                         # v4 : ciblage confusions eval (tarqi3→barber, retouche robe→painter,
                         # rideaux→carpenter) — la COUTURE des rideaux appartient au tailor
                         'tar9i3 ta3 serwal met9atta3 men rokba', 'raccourcir un jean jdid',
                         'coudre des rideaux occultants l chambre', 'changer la fermeture ta3 pantalon',
                         'tadyi9 ta3 9amidja men jnab'],
    },
    'caterer': {
        'ar_names': ['طباخة', 'تريتور', 'مول الطياب'],
        'lat_names': ['traiteur', 'tabbakha', 'cuisiniere'],
        'ar_problems': ['نحتاج طباخة لعرس مية نفر', 'تريتور للختانة نهار الجمعة',
                        'نحب طباخة تجي تطيب في الدار لمناسبة', 'كسكسي لخمسين نفر لصدقة',
                        'نحتاج تريتور لفطور رمضان لعائلة كبيرة', 'طباخة لحفلة النجاح',
                        'نحب نكوموندي حلويات تقليدية للعرس', 'طباخة تطيب الشخشوخة لمناسبة',
                        'نحتاج طياب للوليمة نهار السبت'],
        'lat_problems': ['traiteur pour mariage 100 personnes', 'tabbakha lel 3ers',
                         'couscous l 50 personnes lel sada9a', 'traiteur pour f tour ramadan',
                         'gateaux traditionnels lel 3ers', 'tabbakha tji ttayeb f dar',
                         'buffet pour fete de fiançailles', 'chakhchoukha l mounasba',
                         'repas ta3 khetana jour el jem3a'],
    },
}

WILAYAS = [
    ('الجزائر العاصمة', 'Alger'), ('وهران', 'Oran'), ('قسنطينة', 'Constantine'),
    ('سطيف', 'Setif'), ('عنابة', 'Annaba'), ('بليدة', 'Blida'), ('باتنة', 'Batna'),
    ('تيزي وزو', 'Tizi Ouzou'), ('بجاية', 'Bejaia'), ('جيجل', 'Jijel'),
    ('تلمسان', 'Tlemcen'), ('ورقلة', 'Ouargla'), ('غرداية', 'Ghardaia'),
    ('مستغانم', 'Mostaganem'), ('سكيكدة', 'Skikda'), ('الشلف', 'Chlef'),
    ('بسكرة', 'Biskra'), ('برج بوعريريج', 'Bordj Bou Arreridj'),
    ('تيارت', 'Tiaret'), ('بومرداس', 'Boumerdes'),
]

# ── Gabarits par intent — {name} {problem} {wilaya} ──────────────────────────
T = {
    'find_worker': {
        'ar': [
            'راني نحوس على {name} في {wilaya}',
            'نحتاج {name} مليح في {wilaya}',
            'واش كاين شي {name} قريب من {wilaya}؟',
            'دلوني على {name} شاطر ومعقول',
            'عندي {problem} ونحتاج واحد يصلحهالي',
            '{problem}، شكون يعرف {name} مليح؟',
            'نقدر نلقى {name} يجي اليوم لـ {wilaya}؟',
            'يعطيكم الصحة، نحوس {name} ناشط في {wilaya}',
            'عندي {problem} في الدار، واش نديار؟',
            'شكون عندو نيميرو تاع {name} في {wilaya}؟',
            # style vocal : problème brut, sans nom de métier ni ponctuation
            '{problem}',
            'عندي {problem}',
            '{problem} واش نديار',
            '{problem} عاونوني',
            'خويا {problem} شكون يجي يشوفها',
            '{problem} من فضلكم',
            '{problem} شكون عندو حل',
            'راهي عندي {problem} وحاصل',
            # v4 : récits plus longs / temporalité — style requêtes réelles
            'جاري قالي نسقسي هنا على {name} في {wilaya}',
            'الدار فيها {problem} من سيمانة كاملة',
            '{problem} ومازال ما لقيتش لي يصلحها',
            'نحتاج {name} يجي غدوة الصباح لـ {wilaya}',
        ],
        'lat': [
            'rani n7awes 3la {name} f {wilaya}',
            'je cherche un {name} à {wilaya}',
            'wach kayen {name} 9rib meni f {wilaya}?',
            '3andi {problem}, chkoun ye3ref {name} mli7?',
            'besoin d un {name} serieux f {wilaya}',
            '{problem} ... chkoun ysal7hali?',
            'nheb {name} yji lyoum l {wilaya}',
            'chkoun 3ando numero ta3 {name} f {wilaya}?',
            '{problem}',
            '3andi {problem}',
            '{problem} wach ndir',
            '{problem} 3awnouni',
            '{problem} men fadlkom',
            '{problem} chkoun 3ando solution',
            'depuis 3 jours 3andna {problem}',
            'ma l9itch {name} disponible f {wilaya}',
            '{problem} w ma 3reftch win nrou7',
        ],
    },
    'urgent_service': {
        'ar': [
            'عاونوني! {problem} وراني حاصل',
            'حالة مستعجلة: {problem}',
            '{problem} ونحتاج {name} ديركت توا',
            'بليز {name} في {wilaya} دورك دورك، {problem}',
            'ارجو المساعدة بسرعة، {problem}',
            '{problem}!! نحتاج واحد يجي فيسع',
            'مستعجل: نحتاج {name} اليوم في {wilaya}',
            'الله يخليكم {problem} ومقدرتش نلقى حل',
            'دورك دورك {problem}',
            '{problem} فيسع فيسع',
        ],
        'lat': [
            'urgence!! {problem} f {wilaya}',
            '3awnouni {problem} w rani 7asel',
            '{problem} n7taj {name} tawa dork',
            'svp {name} f {wilaya} vite, {problem}',
            'cas urgent: {problem}',
            '{problem}!! chkoun yji fissa3?',
            '{problem} fissa3 svp',
            'dork dork {problem}',
        ],
    },
    'price_inquiry': {
        'ar': [
            'شحال يدير {name} في {wilaya}؟',
            'شحال السوم تاع {name}؟',
            'شحال نخلص باش نصلح {problem}؟',
            'بشحال الخدمة تاع {name} تقريبا؟',
            'شحال يكلفني {name} لليوم كامل؟',
            'واش هي الأسعار تاع {name} في {wilaya}؟',
            'شحال تكلف خدمة: {problem}؟',
            'شحال يكلف {problem}؟',
        ],
        'lat': [
            'ch7al ydir {name} f {wilaya}?',
            'combien coute un {name}?',
            'ch7al nkhalles bach nsale7 {problem}?',
            'les prix ta3 {name} f {wilaya}?',
            'ch7al ykalefni {name} pour une journee?',
            'devis approximatif: {problem}?',
            'combien pour {problem}?',
            'ch7al taman bach ndir {problem}?',
        ],
    },
    'app_question': {
        'ar': [
            'كيفاش نخدم بهاد التطبيق؟',
            'كيفاش نبعث طلب خدمة؟',
            'واش معنى الاشتراك برو للعامل؟',
            'كيفاش نبدل رقم تيليفوني في الكونت؟',
            'كيفاش نزيد تصاور لخدماتي؟',
            'واش نقدر نخدم كعامل وكليان في نفس الوقت؟',
            'كيفاش نمسح الحساب تاعي؟',
            'وين نلقى الطلبات لي بعثتهم؟',
            'كيفاش نقيّم العامل بعد الخدمة؟',
            'علاش ما يوصلنيش الإشعار كي يجاوبني عامل؟',
            'كيفاش نفعل الموقع باش يلقاوني العمال؟',
            'واش التطبيق مجاني ولا بالخلاص؟',
            'كيفاش نشارك في العروض تاع الخدمات؟',
            'نسيت كلمة السر، واش نديار؟',
            'واش الخلاص يكون كاش ولا بالكارطة؟',
            'واش التطبيق يخدم بلا انترنات؟',
            'كيفاش نبدل المهنة تاعي في البروفيل؟',
            'شحال يكلف الاشتراك في الشهر؟',
            'واش نقدر نستعمل التطبيق كضيف بلا كونت؟',
            'واش نقدر نلغي طلب بعثتو؟',
            'كيفاش نشوف تقييمات العامل قبل ما نخدم معاه؟',
        ],
        'lat': [
            "comment ca marche l'application?",
            'kifach neb3et demande de service?',
            "c'est quoi l'abonnement pro?",
            'kifach nbeddel numero ta3i f compte?',
            'kifach nzid des photos l khedmti?',
            'wach n9der nkoun worker w client f meme temps?',
            'kifach nes7ab compte ta3i?',
            'win nel9a les demandes li b3aththom?',
            'kifach n3ayet l worker apres la mission?',
            "3lach les notifications ma yjounich?",
            "l'appli gratuite wela payante?",
            "j'ai oublié mon mot de passe, wach ndir?",
            'le paiement fel appli kifach ykoun?',
            'wach ne9der nesta3mel l appli offline?',
            'kifach nghayyer profession fel compte ta3i?',
            'ch7al prix ta3 abonnement par mois?',
            'wach ne9der ncancel demande b3aththa?',
            'kifach nchouf les avis ta3 worker?',
            'compte guest kifach ykhdem?',
            'kifach nactivi la localisation fel appli?',
        ],
    },
    'greeting_chitchat': {
        'ar': [
            'سلام', 'سلام عليكم', 'صحا خويا', 'شكرا بزاف', 'واش راك؟',
            'صباح الخير', 'مساء الخير', 'يعطيك الصحة', 'الله يبارك عليكم',
            'مرحبا', 'شكرا على المساعدة', 'ربي يحفظكم', 'صحيت خويا',
            'واش الأحوال؟', 'لاباس عليكم؟', 'تحياتي ليكم', 'بونجور',
            'بالسلامة', 'تصبحو على خير', 'ههههه شابة', 'يعيشك خويا',
        ],
        'lat': [
            'salam', 'slm', 'cc', 'bonjour', 'bonsoir', 'merci khouya',
            'cv?', 'wesh rak?', 'saha khouya', 'merci bcp', 'labas?',
            'salut', 'thanks', 'ya3tik saha', 'chokran',
            'bslema', 'bn nuit khouya', 'hhhh chaba', 'ya3ychek',
        ],
    },
    'out_of_scope': {
        'ar': [
            'وين نلقى بيتزيريا مليحة في {wilaya}؟',
            'شحال راهي الساعة؟',
            'راني نحوس على خدمة نخدمها، واش كاين بلايص؟',
            'واش راهي أخبار الماتش تاع البارح؟',
            'نحب وصفة كسكسي بالدجاج',
            'معليش تقولي كيفاش راه الطقس في {wilaya}؟',
            'نبيع تيليفون سامسونج شبه جديد',
            'وين كاين أقرب صيدلية في {wilaya}؟',
            'شكون ربح الكان هاد العام؟',
            'نحوس على شقة للكراء في {wilaya}',
            'واش من فيلم مليح نتفرجو اليوم؟',
            'كيفاش نديار كونت فيسبوك؟',
            'عندي مشكلة في الويفي تاع دجيزي',
            'نحب نشري طوموبيل مستعملة، واش تنصحوني؟',
            'وين نقرا لانجليزية في {wilaya}؟',
            'التيليفون تاعي تهرس الايكران وين نصلحو؟',
            'البيسي تاعي ولا ثقيل بزاف واش نديار؟',
            'نحب نتعلم الكهرباء واش كاين تكوين؟',
            'وين نلقى تكوين في النجارة؟',
            'نحوس على طبيب أسنان مليح في {wilaya}',
            'نحتاج طاكسي للمطار غدوة الصباح',
            'نحب نشري ماشينة صابون جديدة شحال السوم؟',
            'نحوس أستاذ يعطي دروس خصوصية في الرياضيات',
            'نحب نتعلم الخياطة واش كاين تكوين قريب؟',
            'وين نشري ماكينة خياطة جديدة في {wilaya}؟',
            'نحب نتعلم الحلاقة باش نحل صالون',
            'نحوس على ريسطورون مليح للعشاء في {wilaya}',
            'وين نشري بلاكات بلاكو بالجملة؟',
            'نحوس وصفة طاجين حلو للعيد',
            # v3.1 : jardinage hors catalogue → négatifs durs
            'نحوس جنايني يقلم الأشجار تاع الجنان',
            'الجنان محتاج تنقية وقازون جديد',
            'وين نشري الزرع والورد للجنان في {wilaya}؟',
            'نحتاج واحد يقص الحشيش تاع الجنان',
            'السياج الأخضر تاع الجنان يحتاج تقليم',
            'شكون يركب سقي أوتوماتيك في الجنينة؟',
        ],
        'lat': [
            'win nel9a pizzeria mli7a f {wilaya}?',
            'ch7al rahi el sa3a?',
            'rani n7awes 3la khedma nekhdemha, kayen des postes?',
            'chkoun rbe7 el match lbare7?',
            'recette ta3 couscous svp',
            'kifach rah el jaw f {wilaya}?',
            'nbi3 telephone samsung presque jdid',
            'win kayen pharmacie 9riba f {wilaya}?',
            'n7awes 3la appartement l kra f {wilaya}',
            'kifach ndir compte facebook?',
            '3andi mochkla f wifi ta3 djezzy',
            'wach men film mli7 nchoufou lyoum?',
            'ecran ta3 telephone therres win nsal7ou?',
            'pc ta3i wela lourd bezzaf wach ndir?',
            'nheb net3allem la soudure kayen formation?',
            'n7awes 3la dentiste mli7 f {wilaya}',
            'n7taj taxi l aeroport ghedwa',
            'nheb nechri machine a laver jdida ch7al?',
            'prof ta3 maths l les cours particuliers',
            'nb7ath 3la babysitter l wladi',
            'nheb net3allem la couture kayen des cours?',
            'win nechri machine a coudre f {wilaya}?',
            'restaurant mli7 l le diner f {wilaya}?',
            'win nechri des plaques ba13 en gros?',
            'recette ta3 tajine 7lou lel 3id',
            'n7awes jardinier mli7 ytayye7li les mauvaises herbes',
            'win nechri gazon w les plantes f {wilaya}?',
            'chkoun y9asli el gazon ta3 jardin?',
            'les haies y7tajo taille chkoun ydirha?',
            'arrosage automatique lel jnina chkoun yrakbou?',
        ],
    },
}

# Intents dont les gabarits consomment {name}/{problem} → portent une profession.
PRO_INTENTS = {'find_worker', 'urgent_service', 'price_inquiry'}

# 16 métiers (v3) → cibles relevées pour garder ≥100 lignes/métier.
# v4 : +20% de volume — le LEX élargi (classes faibles) porte la diversité.
TARGETS = {'find_worker': 2300, 'urgent_service': 1150, 'price_inquiry': 1050,
           'app_question': 600, 'greeting_chitchat': 280, 'out_of_scope': 1200}

AR_PREFIX = ['سلام خويا، ', 'سلام، ', 'يا جماعة ', 'من فضلكم ', 'صحا، ',
             'ياخو ', 'يا خويا ', 'صاحبي ', 'يا شيخ ']
LAT_PREFIX = ['slm, ', 'svp ', 'bonjour, ', 'saha, ', 'les amis ',
              'ya kho ', 'ya khouya ', 'sa7bi ', 'ya cheikh ']
AR_SUFFIX = [' الله يخليك', ' وشكرا مسبقا', ' يعطيكم الصحة', ' بارك الله فيكم',
             ' ياخو', ' والله غير حاصل']
LAT_SUFFIX = [' svp', ' merci', ' w chokran', ' rabi y7afdkom',
              ' ya kho', ' wallah ghir 7asel']

# ── Registres régionaux (v3) — swaps lexicaux appliqués APRÈS remplissage ────
# centre = base (gabarits tels quels). Poids ≈ démographie + diversité voulue.
REGION_WEIGHTS = {'center': 40, 'west': 18, 'tlemcen': 10, 'east': 18, 'south': 14}
REGION_SWAPS = {
    'west': {
        'ar':  {'مليح': 'غايا', 'دورك': 'دوك', 'توا': 'دوك', 'نحب': 'نبغي', 'تاع': 'نتاع'},
        'lat': {'mli7': 'ghaya', 'dork': 'douk', 'nheb': 'nebghi', 'ta3': 'nta3'},
    },
    'east': {
        'ar':  {'بزاف': 'ياسر', 'دورك': 'ضركا', 'كيفاش': 'كيفاه'},
        'lat': {'bezzaf': 'yasser', 'dork': 'dorka', 'kifach': 'kifah'},
    },
    'south': {
        'ar':  {'دورك': 'ضروك', 'توا': 'ضروك', 'مليح': 'زين', 'نحب': 'نبغي'},
        'lat': {'dork': 'drouk', 'mli7': 'zin', 'nheb': 'nebghi'},
    },
}


# ── v5 : translittération ar→arabizi (fix terrain «daw magtou3») ─────────────
# Mots fréquents des gabarits → orthographes arabizi réelles (listes = variantes,
# l'inconsistance orthographique du terrain est un signal d'entraînement voulu).
AR2LAT_WORDS = {
    'الضو': ['daw', 'eddaw', 'dou'], 'مقطوع': ['ma9tou3', 'magtou3', 'me9tou3'],
    'الما': ['el ma', 'lma'], 'الدار': ['dar', 'eddar'], 'في': ['f', 'fi'],
    'نحتاج': ['n7taj'], 'نحوس': ['n7awes', 'nhawes'], 'نحب': ['nheb', 'n7eb'],
    'خاصني': ['khasni', 'khassni'], 'واش': ['wach', 'wech'],
    'شحال': ['ch7al', 'chhal'], 'كيفاش': ['kifach', 'kifech'],
    'مليح': ['mli7', 'mlih'], 'بزاف': ['bezzaf', 'bzaf'],
    'يقطر': ['y9atter', 'y9ater'], 'تقطر': ['t9atter'],
    'مسدودة': ['msdouda', 'masdouda'], 'محروقة': ['ma7rou9a', 'mahrouga'],
    'قديم': ['9dim', 'kdim'], 'جديد': ['jdid'], 'جديدة': ['jdida'],
    'كامل': ['kamel'], 'كاملة': ['kamla'], 'شكون': ['chkoun'], 'وين': ['win'],
    'اليوم': ['lyoum'], 'غدوة': ['ghedwa', 'ghodwa'], 'دورك': ['dork'],
    'توا': ['tawa'], 'يطيح': ['yti7'], 'ديما': ['dima'],
    'يشعل': ['yech3el', 'ycha3l'], 'ويطفي': ['w yetfi'], 'وحدو': ['wa7do', 'wahdo'],
    'مول': ['moul', 'mol'], 'سباك': ['sebbak'], 'بلومبيي': ['plombi', 'bloumbi'],
    'كهربائي': ['kahrabji', 'kahrabaji'], 'تريسيان': ['trisyan'],
    'الكليمة': ['la clim', 'clima', 'el klima'],
    'الكليماتيزور': ['climatiseur', 'el klimatizor'],
    'ماشينة': ['machina'], 'الماشينة': ['la machine', 'el machina'],
    'الديجونكتور': ['disjoncteur', 'el dijoncteur'], 'بريزة': ['priza', 'prise'],
    'بريز': ['prise'], 'الحمام': ['el hammam', 'le7mam'],
    'الكوزينة': ['la cuisine', 'el kouzina'], 'اللافابو': ['lavabo', 'el lavabo'],
    'التواليت': ['toilette'], 'الطواليت': ['toilette'],
    'الشوفو': ['chauffe eau', 'el chofo'], 'البالوعة': ['el balou3a'],
    'الروبيني': ['robini', 'el roubini'], 'حيط': ['7it'], 'الحيط': ['el 7it', 'l7it'],
    'السقف': ['plafond', 'es s9af'], 'سقف': ['s9af', 'plafond'],
    'الجنان': ['el jnan', 'jardin'], 'يعاونني': ['y3aweni'], 'يجيني': ['yjini'],
    'نبدل': ['nbeddel'], 'نركب': ['nrakeb'], 'تركيب': ['tarkib'],
    'يصلح': ['ysala7', 'ysalah'], 'تصليح': ['tesli7'], 'خسر': ['khser'],
    'خسرت': ['khesret'], 'ريحة': ['ri7a', 'riha'], 'كاين': ['kayen'],
    'فيسع': ['fissa3'], 'ياخو': ['ya kho'], 'خويا': ['khouya'], 'صحا': ['saha'],
}
AR2LAT_CHARS = {
    'ا': 'a', 'أ': 'a', 'إ': 'i', 'آ': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th',
    'ج': 'j', 'ح': '7', 'خ': 'kh', 'د': 'd', 'ذ': 'd', 'ر': 'r', 'ز': 'z',
    'س': 's', 'ش': 'ch', 'ص': 's', 'ض': 'd', 'ط': 't', 'ظ': 'd', 'ع': '3',
    'غ': 'gh', 'ف': 'f', 'ق': '9', 'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n',
    'ه': 'h', 'ة': 'a', 'ى': 'a', 'ء': '', 'ئ': 'i', 'ؤ': 'ou', 'ّ': '',
    '،': ',', '؟': '?', 'ـ': '',
}


def _translit_word(w: str) -> str:
    if w in AR2LAT_WORDS:
        return random.choice(AR2LAT_WORDS[w])
    if w.startswith('ال') and len(w) > 3:  # article : el/l + reste (re-teste le dico)
        return random.choice(['el ', 'l']) + _translit_word(w[2:])
    toks = []
    for i, c in enumerate(w):
        if c == 'و':
            toks.append('w' if i == 0 else 'ou')
        elif c == 'ي':
            toks.append('y' if i == 0 else 'i')
        else:
            toks.append(AR2LAT_CHARS.get(c, c))
    # l'arabe n'écrit pas les voyelles brèves → brise les grappes de 3 consonnes
    out, run = [], 0
    for t in toks:
        if t and not (set(t) & set('aeiou')):
            run += 1
            if run == 3:
                out.append('e')
                run = 1
        else:
            run = 0
        out.append(t)
    return ''.join(out)


def ar2lat(text: str) -> str:
    """Ligne arabe entière → arabizi plausible ; le latin passe inchangé."""
    words = []
    for w in text.split(' '):
        pre = suf = ''
        while w and w[0] in '،؟!.,?':
            pre += AR2LAT_CHARS.get(w[0], w[0])
            w = w[1:]
        while w and w[-1] in '،؟!.,?':
            suf = AR2LAT_CHARS.get(w[-1], w[-1]) + suf
            w = w[:-1]
        core = _translit_word(w) if any('\u0600' <= c <= '\u06ff' for c in w) else w
        words.append(pre + core + suf)
    return ' '.join(words)


def regionalize(text: str, script: str, region: str) -> str:
    """Applique le registre régional. Tlemcen = qaf→hamza (قهوة→اهوة, 9→2)."""
    if region == 'tlemcen':
        if script == 'ar' or 'ق' in text:
            out = []
            for w in text.split(' '):
                if w.startswith('ق'):
                    w = 'ا' + w[1:]
                out.append(w.replace('ق', 'أ'))
            text = ' '.join(out)
        return text.replace('9', '2')
    swaps = REGION_SWAPS.get(region, {}).get(script if script in ('ar', 'lat') else 'lat', {})
    # ponytail: remplacement borné par espaces — rate les mots collés à la
    # ponctuation, suffisant comme signal d'entraînement.
    padded = f' {text} '
    for a, b in swaps.items():
        padded = padded.replace(f' {a} ', f' {b} ').replace(f' {a}،', f' {b}،')
    return padded.strip()


def qaf_axis(text: str, script: str) -> str:
    """Axe orthographique du qaf : ar ق→{ڨ,گ}, arabizi 9→{q,g}."""
    if script == 'ar':
        return text.replace('ق', random.choice(['ڨ', 'گ']))
    return text.replace('9', random.choice(['q', 'g']))


def noisify(text: str, script: str) -> str:
    """Perturbations légères — simule l'orthographe libre du darija réel."""
    r = random.random()
    if r < 0.25:
        text = text.rstrip('؟?!.')
    elif r < 0.45:
        text = random.choice(AR_PREFIX if script == 'ar' else LAT_PREFIX) + text
    elif r < 0.60:
        text = text + random.choice(AR_SUFFIX if script == 'ar' else LAT_SUFFIX)
    elif r < 0.70 and script == 'ar':
        text = text.replace('ة', 'ه')
    elif r < 0.80:
        text = text.replace('بزاف', 'بزااااف').replace('!!', '!!!').replace('bezzaf', 'bzzzaf')
    return text


def fill(template: str, script: str, prof: str | None):
    """Remplit les slots; 15% de mélange trans-script (darija réel = code-switching)."""
    cross = random.random() < 0.15
    other = 'lat' if script == 'ar' else 'ar'
    out = template
    if '{name}' in out:
        src = LEX[prof][f'{other if cross else script}_names']
        out = out.replace('{name}', random.choice(src))
    if '{problem}' in out:
        src = LEX[prof][f'{script}_problems']
        out = out.replace('{problem}', random.choice(src))
    if '{wilaya}' in out:
        ar, lat = random.choice(WILAYAS)
        w = (lat if script == 'lat' else ar) if not cross else (ar if script == 'lat' else lat)
        out = out.replace('{wilaya}', w)
    return out


def generate() -> list[dict]:
    rows, seen = [], set()
    for intent, target in TARGETS.items():
        made, attempts = 0, 0
        while made < target and attempts < target * 60:
            attempts += 1
            script = random.choice(['ar', 'lat'])  # eval réel penche latin → 50/50
            # v5 : 35% des lignes latines = ligne AR entière translittérée en
            # arabizi — installe les mots darja romanisés (daw, magtou3…).
            translit = script == 'lat' and random.random() < 0.35
            tscript = 'ar' if translit else script
            template = random.choice(T[intent][tscript])
            if intent in PRO_INTENTS:
                # ~7% de requêtes vagues sans métier identifiable → none
                if random.random() < 0.07 and '{problem}' not in template and '{name}' in template:
                    generic = {'ar': 'واحد يعاونني في خدمة في الدار',
                               'lat': 'wa7ed y3aweni f khedma f dar'}[tscript]
                    text = fill(template.replace('{name}', generic), tscript,
                                random.choice(list(LEX)))
                    prof_label = 'none'
                else:
                    prof = random.choice(list(LEX))
                    text = fill(template, tscript, prof)
                    prof_label = prof
            else:
                prof = random.choice(list(LEX))  # certains gabarits OOS ont {wilaya} seulement
                text = fill(template, tscript, prof)
                prof_label = 'none'
            if translit:
                text = ar2lat(text)
            if random.random() < 0.45:
                text = noisify(text, script)
            # v3 : registre régional puis axe qaf (hors Tlemcen qui fixe déjà 9→2)
            region = random.choices(list(REGION_WEIGHTS), weights=REGION_WEIGHTS.values())[0]
            text = regionalize(text, script, region)
            if region != 'tlemcen' and random.random() < 0.25:
                text = qaf_axis(text, script)
            text = text.strip()
            if text and text not in seen:
                seen.add(text)
                rows.append({'text': text, 'intent': intent,
                             'profession': prof_label, 'source': 'synth'})
                made += 1
    random.shuffle(rows)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = generate()

    # ── Auto-vérifications (échouent bruyamment si la logique casse) ──────────
    assert len(rows) >= 6000, f'trop peu de lignes: {len(rows)}'
    assert len({r["text"] for r in rows}) == len(rows), 'doublons détectés'
    assert all(r['intent'] in INTENTS and r['profession'] in PROFESSIONS for r in rows)
    by_prof = Counter(r['profession'] for r in rows if r['profession'] != 'none')
    for p in LEX:
        assert by_prof[p] >= 100, f'profession sous-représentée: {p}={by_prof[p]}'
    by_intent = Counter(r['intent'] for r in rows)
    for i in INTENTS:
        assert by_intent[i] >= 150, f'intent sous-représenté: {i}={by_intent[i]}'
    # v3 : les marqueurs régionaux / axe qaf doivent réellement apparaître
    joined = ' '.join(r['text'] for r in rows)
    for marker in ['ڨ', 'گ', 'ياسر', 'ghaya', 'nebghi', 'ياخو', 'ya kho']:
        assert marker in joined, f'marqueur régional absent: {marker}'
    # v5 : l'axe translittération doit réellement produire du darja romanisé
    assert any(m in joined for m in ('ma9tou3', 'magtou3', 'me9tou3')), \
        'axe translit absent (مقطوع)'
    assert any(f' {m} ' in joined for m in ('daw', 'eddaw', 'dou')), \
        'axe translit absent (الضو)'
    assert 'y9atter' in joined or 'y9ater' in joined, 'axe translit absent (يقطر)'

    # eval_heldout.csv est intouchable : zéro fuite train↔eval
    eval_path = OUT / 'eval_heldout.csv'
    if eval_path.exists():
        with eval_path.open() as f:
            eval_texts = {r['text'] for r in csv.DictReader(f)}
        leak = eval_texts & {r['text'] for r in rows}
        assert not leak, f'fuite train/eval: {sorted(leak)[:5]}'

    (OUT / 'labels.json').write_text(json.dumps(
        {'intents': INTENTS, 'professions': PROFESSIONS}, ensure_ascii=False, indent=2))
    with (OUT / 'synth_v5.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['text', 'intent', 'profession', 'source'])
        w.writeheader()
        w.writerows(rows)

    print(f'✓ {len(rows)} lignes → {OUT / "synth_v5.csv"}')
    print('intents  :', dict(by_intent))
    print('métiers  :', dict(by_prof))


if __name__ == '__main__':
    main()
