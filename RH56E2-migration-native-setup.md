# Grove-G1 — Dex3-1'den Inspire RH56E2'ye Geçiş ve Native Kurulum

Bu belge, `grove-g1` (`humble-unitree` branch) stack'indeki Unitree Dex3-1 ellerinin
Inspire RH56E2 ile değiştirilmesi sürecini ve bu değişikliğin Docker kullanmadan,
Ubuntu 22.04 + ROS 2 Humble üzerinde nasıl devreye alındığını anlatır.

> **Referans repo:** `https://github.com/Mertsr/g1-dex5-mujoco` (RH56E2 değişikliklerinin
> tutulduğu kopya — orijinal `Adyansh04/grove-g1`'in `humble-unitree` dalından türetildi)
> **Hedef makine:** Ubuntu 22.04, ROS 2 Humble, CycloneDDS, `~/projects/grove-g1`
> **Tarih:** 10 Ekim 2026

---

## 1. Neden bu değişiklik

G1'in standart Dex3-1 eli (7 eklem: thumb_0/1/2, middle_0/1, index_0/1) yerine
Inspire RH56E2 eli (12 eklem — 6 sürülen + 6 mimic) kullanılmak istendi. Kapsam:

- URDF + xacro
- MoveIt 2 (SRDF, controller, joint limits)
- `ros2_control` hand interface plugin'i
- MuJoCo simülatörü (`unitree_mujoco`)

Gerçek donanım sürücüsü (Modbus/seri haberleşme) bu kapsamın dışında — wire
sözleşmesi (`/rh56e2/<side>/{cmd,state}` DDS topic'leri) hazırlandı ama karşı
tarafı (gerçek eli konuşan node) henüz yazılmadı.

---

## 2. RH56E2 elinin yapısı

| | Dex3-1 | RH56E2 |
|---|---|---|
| Toplam eklem | 7 | 12 (6 sürülen + 6 mimic) |
| Sürülen eklemler | thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1 | thumb_1, thumb_2, index_1, middle_1, ring_1, little_1 |
| Mimic eklemler | yok | thumb_3→thumb_4 (0.8392), thumb_2→thumb_3 zinciri sadeleştirildi → thumb_2→thumb_4 (0.7477), index/middle/ring/little `_1`→`_2` |
| Kütle (gerçek) | ~0.3 kg | 0.777 kg/el |
| Kütle (sim'de) | ~0.3 kg | 0.170 kg/el (ölçekli — ağırlığın gerçeği değil!) |
| DDS topic'leri | `rt/dex3/<side>/{cmd,state}` | `rt/rh56e2/<side>/{cmd,state}` |

Her eklemin aralığı 0'dan başlıyor ve pozitife kapanıyor (iki el için de aynı yön).

---

## 3. Değişen dosyalar (workspace/ içinde)

### `g1_description`
- `urdf/g1_29dof_no_hands.urdf` — vendor URDF'den Dex3 parmak link/joint'leri
  çıkarılmış hâli (`scripts/make_no_hands_urdf.py` ile üretildi).
- `urdf/rh56e2_left.urdf.xacro`, `urdf/rh56e2_right.urdf.xacro` — vendor'un
  SolidWorks export'undan türetilmiş xacro makroları.
- `urdf/g1_arm_sdk.urdf.xacro` — no-hands URDF'yi + iki RH56E2 xacro'yu birleştirip
  `ros2_control` bloklarını (`G1HandSystemLeft/Right`) tanımlıyor.
- `config/rh56e2_params.yaml` — eklem limitleri, kp/kd (placeholder), montaj açısı,
  grasp frame konumu. Eski `dex3_params.yaml` silindi.
- `meshes/rh56e2/{left,right}/` — vendor STL'leri (gerçek mesh'ler sizin tarafınızdan
  eklendi; depoda sadece test amaçlı placeholder'lar vardı).

### `g1_hand_interface`
- `g1_dex3_system.{hpp,cpp}` → `g1_hand_system.{hpp,cpp}` olarak yeniden adlandırıldı.
- `kNumHandJoints` 7'den 6'ya düştü, eklem sırası RH56E2'nin sürülen eklemleri.
- Topic adları `/dex3/...` → `/rh56e2/...`.

### `g1_moveit_config`
- `config/g1.srdf` — `left_hand`/`right_hand` planlama grupları RH56E2'nin 12
  eklemini (6 sürülen + 6 mimic) listeliyor; `open`/`closed` duruşları yeniden
  tanımlandı; çarpışma matrisi (`disable_collisions`) elin kendi içi ve
  wrist_yaw'a karşı yeniden yazıldı.
- `config/joint_limits.yaml`, `config/moveit_controllers.yaml` — 6 sürülen eklem.

### `g1_bringup`
- `config/controllers.yaml` — hand controller joint listeleri.
- `config/sim_sensors.yaml` — `grasp_weld.capture_radius_m` 0.12 → 0.20 (yeni grasp
  frame avuçtan daha uzakta), kapanma oranı yorumu 12 ekleme göre güncellendi.
- `scripts/activate_arm`, `scripts/deactivate_arm` — yorum güncellemeleri.

### `vendor/unitree_mujoco` + `patches/unitree_mujoco`
- `dex3_handler.cc/.h` — 7 değil 6 eklem okuyup yazıyor, `rt/rh56e2/<side>/...`
  DDS topic'lerini açıyor.
- `grasp_weld.cc/.h` — kapanma oranı artık 12 eklemin ortalaması (gerçek `closed`
  duruşunda ölçülen değer: 0.71).
- `003-add-dex3-hands.patch` silindi, yerine `003-add-rh56e2-hands.patch` geldi.
  Bu patch **elle yazılmıyor** — `tools/make_rh56e2_patch.py`, genişletilmiş
  URDF'yi okuyup MJCF'ye çeviriyor, böylece sim ve URDF hiç birbirinden sapmıyor.

---

## 4. Native (Docker'sız) kurulum — yapılan adımlar

Mevcut native kurulum belgenizdeki (`GroveG1-humble_knowlage_and_cmd.md`) yolları
esas alındı: `~/projects/grove-g1`, `/opt/unitree_robotics`, MuJoCo 3.3.6.

### 4.1 Workspace dosyalarını senkronize etme

```bash
cd ~/projects/grove-g1
git clone https://github.com/Mertsr/g1-dex5-mujoco.git /tmp/g1-dex5-mujoco-ref
```

Ardından referans depodan `workspace/src/{g1_description,g1_hand_interface,
g1_moveit_config,g1_bringup,g1_manipulation,g1_orchestration,g1_motion_service_sim}`
ve `workspace/vendor/unitree_mujoco`, `workspace/patches/unitree_mujoco` altındaki
ilgili dosyalar kopyalandı (tam liste: repo'daki commit geçmişi).

**Önemli not:** `g1_description/meshes/rh56e2/` **kopyalanmadı** — sizde gerçek
vendor STL'leri olduğu için referans depodaki placeholder'larla üzerine yazılmadı.

Dikkat edilen iki bash tuzağı:
- `cp src/{tek_dosya}` — tek elemanlı süslü parantez bash'te açılmaz, parantezler
  kaldırıldı.
- Hedef dizin yoksa `cp ... dest/` "Not a directory" hatası verir — her kopyalama
  öncesi `mkdir -p` eklendi.

### 4.2 Simülatörü sıfırdan yeniden kurma

Eski `/opt/unitree_robotics/unitree_mujoco` zaten Dex3 patch'leriyle derlenmiş
olduğundan, üzerine RH56E2 patch'i uygulanamaz — sıfırdan başlandı:

```bash
# Eskiyi yedekle (silme)
sudo mv /opt/unitree_robotics/unitree_mujoco /opt/unitree_robotics/unitree_mujoco.dex3-backup

# Pinned SHA'dan temiz klon
cd /tmp && rm -rf unitree_mujoco
git clone https://github.com/unitreerobotics/unitree_mujoco.git
cd unitree_mujoco
git checkout ae6a8403e272733e9996ef59990880330496177f

# Patch zinciri (001 → 002 → 003-add-rh56e2-hands → 004 → 005 → 006)
for patch in ~/projects/grove-g1/workspace/patches/unitree_mujoco/*.patch; do
  [ -e "$patch" ] || continue
  git apply --check "$patch" && git apply "$patch"
done

# Handler kaynakları
cp ~/projects/grove-g1/workspace/vendor/unitree_mujoco/*.h simulate/src/
cp ~/projects/grove-g1/workspace/vendor/unitree_mujoco/*.cc simulate/src/

# RH56E2 mesh'leri (gerçek STL'ler)
mkdir -p unitree_robots/g1/meshes/rh56e2
cp -r ~/projects/grove-g1/workspace/src/g1_description/meshes/rh56e2/{left,right} \
      unitree_robots/g1/meshes/rh56e2/

# /opt altına taşı
sudo cp -a /tmp/unitree_mujoco /opt/unitree_robotics/unitree_mujoco
sudo chown -R "$USER:$USER" /opt/unitree_robotics/unitree_mujoco

# MuJoCo symlink'i
cd /opt/unitree_robotics/unitree_mujoco
rm -rf simulate/mujoco
ln -s /opt/unitree_robotics/mujoco-3.3.6 simulate/mujoco

# Derle
rm -rf simulate/build && mkdir -p simulate/build && cd simulate/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/unitree_robotics/lib/cmake
cmake --build . --target unitree_mujoco --parallel 4
```

### 4.3 Hangi robotun yükleneceğini seçme (`config.yaml`)

`unitree_mujoco` binary'si komut satırı argümanı olarak dosya adı almıyor —
hangi robotu/sahneyi yükleyeceğini **`simulate/config.yaml`**'dan okuyor. İlk
denemede varsayılan `robot: "go2"` olduğu için binary bir dört ayaklı robot
(köpek) yükledi, G1 değil:

```bash
find /opt/unitree_robotics/unitree_mujoco -iname "config.yaml"
# -> /opt/unitree_robotics/unitree_mujoco/simulate/config.yaml
```

`robot` ve `robot_scene` alanları G1'e çevrildi:
```yaml
robot: "g1"
robot_scene: "scene_29dof.xml"
```

Sonrasında binary doğru şekilde G1'i, RH56E2 elleriyle yükledi.

### 4.4 ROS workspace'i derleme

```bash
cd ~/projects/grove-g1/workspace
source /opt/ros/humble/setup.bash
MAKEFLAGS=-j4 colcon build --symlink-install --parallel-workers 4 \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

**Sonuç: 19 paket, hata yok.** `g1_description` ve `g1_hand_interface` sadece
zararsız uyarılar verdi (lint araçları kurulu değil, ROS'un kendi deprecated
header notu) — gerçek bir derleme hatası çıkmadı.

---

## 5. Doğrulama

```bash
source install/setup.bash
ros2 launch g1_bringup bringup.launch.py headless:=false
```

Başka terminalde:
```bash
ros2 topic list | grep rh56e2          # /rh56e2/left|right/{cmd,state}
ros2 topic echo /joint_states --once   # her elde 6 eklem (7 değil)
ros2 control list_hardware_components  # G1HandSystemLeft / G1HandSystemRight
```

---

## 6. Hâlâ doğrulanmamış / bilinmesi gerekenler

- **Montaj rotasyonu ve grasp frame konumu** (`rh56e2_params.yaml`): vendor'un
  eksen yönlerinden tahmin edildi, RViz'de gözle doğrulanmadı.
- **`closed` duruşu ve SRDF çarpışma çiftleri**: tahmin, gerçek `move_group`
  ile test edilmedi.
- **kp/kd kazançları ve eklem moment sınırı**: placeholder değerler, gerçek
  hand datasheet'i yok.
- **Yürüme politikası**: el kütlesi sim'de 0.170 kg'a ölçeklendi (gerçeği 0.777
  kg) — daha ağır/uzun bir el yürüyüşü etkileyebilir, yeniden test edilmedi.
- **Gerçek donanım sürücüsü**: `/rh56e2/<side>/{cmd,state}` wire sözleşmesi
  hazır, ama bunu gerçek RH56E2'yle (Modbus/seri) konuşturan node henüz yazılmadı.

---

## 7. Geri alma (rollback)

Bir şey ters giderse:
```bash
sudo rm -rf /opt/unitree_robotics/unitree_mujoco
sudo mv /opt/unitree_robotics/unitree_mujoco.dex3-backup /opt/unitree_robotics/unitree_mujoco
cd ~/projects/grove-g1 && git stash   # workspace/src değişikliklerini geri al (stash alınmışsa)
```
