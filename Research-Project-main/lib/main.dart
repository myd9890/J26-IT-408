import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

const apiBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'http://127.0.0.1:8000',
);

class ScanResult {
  final Map<String, dynamic> data;
  const ScanResult(this.data);
  String? get predictedClass => data['predicted_class'] as String?;
  String get status => (data['status'] ?? 'unknown') as String;
  String? get message => data['message'] as String?;
  double? get confidence => (data['model_confidence_percentage'] as num?)?.toDouble();
  String? get originalImage => data['original_image_base64'] as String?;
  String? get overlayImage => data['gradcam_overlay_base64'] as String?;
  String? get explanationTarget => data['explanation_target'] as String?;
  Map<String, dynamic> get classScores => Map<String, dynamic>.from(data['class_scores'] ?? {});
}

class Component1Api {
  static Future<ScanResult> scan(XFile file) async {
    final request = http.MultipartRequest('POST', Uri.parse('$apiBaseUrl/api/v1/scans'));
    request.files.add(http.MultipartFile.fromBytes('image', await file.readAsBytes(), filename: file.name));
    final response = await request.send().timeout(const Duration(minutes: 3));
    final body = await response.stream.bytesToString();
    final decoded = jsonDecode(body) as Map<String, dynamic>;
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception(decoded['detail'] ?? 'The disease scan failed.');
    }
    return ScanResult(decoded);
  }

  static Future<List<ScanResult>> history() async {
    final response = await http.get(Uri.parse('$apiBaseUrl/api/v1/scans'));
    if (response.statusCode != 200) throw Exception('Could not load scan history.');
    final rows = jsonDecode(response.body) as List<dynamic>;
    return rows.map((row) => ScanResult(Map<String, dynamic>.from(row))).toList();
  }
}

String displayDisease(String value) => value
    .split('_')
    .map((part) => part.isEmpty ? part : '${part[0].toUpperCase()}${part.substring(1)}')
    .join(' ');

ImageProvider? base64Image(String? value) => value == null || value.isEmpty
    ? null
    : MemoryImage(base64Decode(value));

void main() => runApp(const AgriLeafApp());

class C {
  static const bg = Color(0xfff4f6ef),
      white = Colors.white,
      mint = Color(0xffe4efe3),
      green = Color(0xff155c45),
      deep = Color(0xff092f25),
      ink = Color(0xff14251d),
      muted = Color(0xff6b776d),
      red = Color(0xffc71e27),
      orange = Color(0xffffb657),
      sand = Color(0xffffedd2),
      lime = Color(0xffc8ef83),
      border = Color(0xffdce4d9);
}

class AgriLeafApp extends StatelessWidget {
  const AgriLeafApp({super.key});
  @override
  Widget build(BuildContext c) => MaterialApp(
    debugShowCheckedModeBanner: false,
    title: 'AgriLeaf AI',
    theme: ThemeData(
      useMaterial3: true,
      scaffoldBackgroundColor: C.bg,
      colorScheme: ColorScheme.fromSeed(seedColor: C.green),
      fontFamily: 'Arial',
      textTheme: const TextTheme(
        titleLarge: TextStyle(
          fontSize: 22,
          fontWeight: FontWeight.w700,
          color: C.ink,
        ),
        bodyLarge: TextStyle(fontSize: 15, height: 1.45, color: C.ink),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: C.mint,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(18),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(18),
          borderSide: const BorderSide(color: C.border),
        ),
      ),
    ),
    home: const SplashScreen(),
  );
}

class Logo extends StatelessWidget {
  final double size;
  const Logo({super.key, this.size = 40});
  @override
  Widget build(BuildContext c) => Container(
    width: size,
    height: size,
    padding: EdgeInsets.all(size * .07),
    decoration: BoxDecoration(
      color: C.lime,
      borderRadius: BorderRadius.circular(size * .13),
      boxShadow: const [
        BoxShadow(
          color: Color(0x18000000),
          blurRadius: 14,
          offset: Offset(0, 6),
        ),
      ],
    ),
    child: Image.asset('assets/images/agrileaf_logo.png', fit: BoxFit.contain),
  );
}

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});
  @override
  State<SplashScreen> createState() => _SplashState();
}

class _SplashState extends State<SplashScreen> {
  Timer? timer;
  @override
  void initState() {
    super.initState();
    timer = Timer(const Duration(milliseconds: 1600), () {
      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => const LoginScreen()),
        );
      }
    });
  }

  @override
  void dispose() {
    timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext c) => Scaffold(
    backgroundColor: C.deep,
    body: SafeArea(
      child: Stack(
        children: [
          Positioned(top: -80, right: -70, child: _Orb(220, C.green)),
          Positioned(bottom: -110, left: -60, child: _Orb(250, Color(0xff164b37))),
          Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(color: C.lime.withOpacity(.12), shape: BoxShape.circle),
                  child: const Logo(size: 118),
                ),
                const SizedBox(height: 26),
                const Text('AgriLeaf AI', style: TextStyle(fontSize: 25, fontWeight: FontWeight.w800, color: Colors.white)),
                const SizedBox(height: 10),
                const Text(
                  'Edge AI for Precision Agriculture',
                  style: TextStyle(fontSize: 14, color: Color(0xffb9d6bd)),
                ),
              ],
            ),
          ),
          Positioned(
            left: 42,
            right: 42,
            bottom: 66,
            child: Column(
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: const LinearProgressIndicator(
                    value: .3,
                    minHeight: 4,
                    color: C.lime,
                    backgroundColor: Color(0xff285340),
                  ),
                ),
                const SizedBox(height: 14),
                const Text(
                  'INITIALIZING SENSORS...',
                  style: TextStyle(letterSpacing: 1.5, color: Color(0xffb9d6bd), fontSize: 11),
                ),
              ],
            ),
          ),
        ],
      ),
    ),
  );
}

class _Orb extends StatelessWidget {
  final double size; final Color color;
  const _Orb(this.size, this.color);
  @override
  Widget build(BuildContext context) => Container(width: size, height: size, decoration: BoxDecoration(color: color.withOpacity(.38), shape: BoxShape.circle));
}

class Field extends StatelessWidget {
  final String label, hint;
  final IconData icon;
  final bool obscure;
  const Field(
    this.label,
    this.hint,
    this.icon, {
    super.key,
    this.obscure = false,
  });
  @override
  Widget build(BuildContext c) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
      const SizedBox(height: 8),
      TextField(
        obscureText: obscure,
        decoration: InputDecoration(
          hintText: hint,
          prefixIcon: Icon(icon, color: C.muted),
          suffixIcon: obscure
              ? const Icon(Icons.visibility_outlined, color: C.muted)
              : null,
        ),
      ),
    ],
  );
}

class Primary extends StatelessWidget {
  final String text;
  final VoidCallback tap;
  final Color color;
  const Primary(this.text, this.tap, {super.key, this.color = C.green});
  @override
  Widget build(BuildContext c) => SizedBox(
    width: double.infinity,
    height: 56,
    child: FilledButton(
      style: FilledButton.styleFrom(
        backgroundColor: color,
        foregroundColor: color == C.mint ? C.ink : Colors.white,
        elevation: color == C.green ? 5 : 0,
        shadowColor: C.green.withOpacity(.28),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      ),
      onPressed: tap,
      child: Text(text, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w800, letterSpacing: .2)),
    ),
  );
}

class LoginScreen extends StatelessWidget {
  const LoginScreen({super.key});
  @override
  Widget build(BuildContext c) => Scaffold(
    body: SafeArea(
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const SizedBox(height: 70),
          const Logo(size: 128),
          const Center(child: Text('AgriLeaf AI')),
          const Center(child: Text('Edge AI for Plantation Management')),
          const SizedBox(height: 50),
          const Field('Email or Worker ID', 'Enter ID', Icons.badge_outlined),
          const SizedBox(height: 20),
          const Field(
            'Password',
            'Enter password',
            Icons.lock_outline,
            obscure: true,
          ),
          const SizedBox(height: 30),
          Primary(
            'Sign In',
            () => Navigator.pushReplacement(
              c,
              MaterialPageRoute(builder: (_) => const MainShell()),
            ),
          ),
          TextButton(
            onPressed: () => Navigator.push(
              c,
              MaterialPageRoute(builder: (_) => const SignUpScreen()),
            ),
            child: const Text('Create Account'),
          ),
        ],
      ),
    ),
  );
}

class SignUpScreen extends StatelessWidget {
  const SignUpScreen({super.key});
  @override
  Widget build(BuildContext c) => Scaffold(
    body: SafeArea(
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Logo(size: 96),
          const Center(
            child: Text(
              'AgriLeaf AI',
              style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold),
            ),
          ),
          const Center(child: Text('Create your professional account')),
          const SizedBox(height: 24),
          const Field('Full Name', 'Jane Doe', Icons.person_outline),
          const SizedBox(height: 12),
          const Field(
            'Worker ID or Email',
            'W-12345 or jane@agrileaf.com',
            Icons.badge_outlined,
          ),
          const SizedBox(height: 12),
          const Field(
            'Department',
            'Select Department',
            Icons.apartment_outlined,
          ),
          const SizedBox(height: 12),
          const Field(
            'Password',
            '••••••••',
            Icons.lock_outline,
            obscure: true,
          ),
          const SizedBox(height: 12),
          const Field(
            'Confirm Password',
            '••••••••',
            Icons.lock_clock_outlined,
            obscure: true,
          ),
          const SizedBox(height: 20),
          Primary(
            'Create Account',
            () => Navigator.pushAndRemoveUntil(
              c,
              MaterialPageRoute(builder: (_) => const MainShell()),
              (_) => false,
            ),
          ),
        ],
      ),
    ),
  );
}

class V extends StatelessWidget {
  final double n;
  const V(this.n, {super.key});
  @override
  Widget build(BuildContext c) => SizedBox(height: n);
}

class H extends StatelessWidget {
  final double n;
  const H(this.n, {super.key});
  @override
  Widget build(BuildContext c) => SizedBox(width: n);
}

class Box extends StatelessWidget {
  final Widget child;
  final Color color;
  final VoidCallback? tap;
  const Box(this.child, {super.key, this.color = C.mint, this.tap});
  @override
  Widget build(BuildContext c) => Card(
    margin: EdgeInsets.zero,
    elevation: 0,
    shadowColor: const Color(0x16092f25),
    color: color,
    clipBehavior: Clip.antiAlias,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(22),
      side: color == C.white ? const BorderSide(color: C.border) : BorderSide.none,
    ),
    child: InkWell(
      onTap: tap,
      child: Padding(padding: const EdgeInsets.all(18), child: child),
    ),
  );
}

class Section extends StatelessWidget {
  final String text;
  final Widget? trailing;
  const Section(this.text, {super.key, this.trailing});
  @override
  Widget build(BuildContext c) => Row(
    children: [
      Expanded(
        child: Text(
          text,
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
      if (trailing case final Widget value) value,
    ],
  );
}

void push(BuildContext c, Widget p) =>
    Navigator.push(c, MaterialPageRoute(builder: (_) => p));
void notice(BuildContext c, String s) =>
    ScaffoldMessenger.of(c).showSnackBar(SnackBar(content: Text(s)));

class Header extends StatelessWidget {
  final String title;
  final bool back;
  const Header(this.title, {super.key, this.back = false});
  @override
  Widget build(BuildContext c) => Container(
    height: 82,
    padding: const EdgeInsets.fromLTRB(20, 12, 20, 10),
    color: C.bg,
    child: Row(
      children: [
        back
            ? IconButton(
                onPressed: () => Navigator.pop(c),
                icon: const Icon(Icons.arrow_back_ios_new, color: C.green),
              )
            : const Logo(),
        const H(14),
        Expanded(child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('GOOD MORNING, FIELD LEAD', style: TextStyle(fontSize: 9, letterSpacing: 1.2, color: C.muted, fontWeight: FontWeight.w700)),
            const SizedBox(height: 3),
            Text(title, style: TextStyle(fontSize: 21, fontWeight: FontWeight.w800, color: C.ink)),
          ],
        )),
        Container(
          padding: const EdgeInsets.all(3),
          decoration: BoxDecoration(color: C.white, shape: BoxShape.circle, border: Border.all(color: C.border)),
          child: Stack(
          children: [
            const CircleAvatar(
              radius: 18,
              backgroundImage: AssetImage('assets/images/profile.png'),
            ),
            Positioned(
              right: 0,
              bottom: 0,
              child: Container(
                width: 9,
                height: 9,
                decoration: BoxDecoration(
                  color: Colors.green,
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white, width: 2),
                ),
              ),
            ),
          ],
          ),
        ),
      ],
    ),
  );
}

class Page extends StatelessWidget {
  final String title;
  final List<Widget> children;
  final bool back;
  const Page(this.title, this.children, {super.key, this.back = false});
  @override
  Widget build(BuildContext c) => Column(
    children: [
      Header(title, back: back),
      Expanded(
        child: ListView(
          physics: const BouncingScrollPhysics(),
          padding: const EdgeInsets.fromLTRB(20, 4, 20, 34),
          children: children,
        ),
      ),
    ],
  );
}

class Detail extends StatelessWidget {
  final String title;
  final List<Widget> children;
  const Detail(this.title, this.children, {super.key});
  @override
  Widget build(BuildContext c) => Scaffold(
    body: SafeArea(bottom: false, child: Page(title, children, back: true)),
    bottomNavigationBar: BottomNav(
      index: -1,
      onTap: (i) => Navigator.pushAndRemoveUntil(
        c,
        MaterialPageRoute(builder: (_) => MainShell(initial: i)),
        (_) => false,
      ),
    ),
  );
}

class BottomNav extends StatelessWidget {
  final int index;
  final ValueChanged<int> onTap;
  const BottomNav({super.key, required this.index, required this.onTap});
  static const items = [
    ('Home', Icons.home_rounded),
    ('Map', Icons.grid_view_rounded),
    ('Scan', Icons.center_focus_strong),
    ('Risk', Icons.bolt_rounded),
    ('More', Icons.more_horiz_rounded),
  ];
  @override
  Widget build(BuildContext c) => SafeArea(
    top: false,
    child: Container(
      height: 74,
      decoration: const BoxDecoration(
        color: C.deep,
        borderRadius: BorderRadius.vertical(top: Radius.circular(26)),
      ),
      child: Row(
        children: List.generate(
          5,
          (i) => Expanded(
            child: InkWell(
              onTap: () => onTap(i),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 180),
                    padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 5),
                    decoration: BoxDecoration(color: i == index ? C.lime : Colors.transparent, borderRadius: BorderRadius.circular(14)),
                    child: Icon(items[i].$2, size: 21, color: i == index ? C.deep : const Color(0xffb7d0ba)),
                  ),
                  Text(
                    items[i].$1,
                    style: TextStyle(
                      fontSize: 12,
                      color: i == index ? C.lime : const Color(0xffb7d0ba),
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    ),
  );
}

class MainShell extends StatefulWidget {
  final int initial;
  const MainShell({super.key, this.initial = 0});
  @override
  State<MainShell> createState() => _ShellState();
}

class _ShellState extends State<MainShell> {
  late int index = widget.initial;
  static const pages = [
    FarmOverview(),
    FieldMap(),
    PlantScan(),
    LiveRisk(),
    MorePage(),
  ];
  @override
  Widget build(BuildContext c) => Scaffold(
    body: SafeArea(bottom: false, child: pages[index]),
    bottomNavigationBar: BottomNav(
      index: index,
      onTap: (v) => setState(() => index = v),
    ),
  );
}

class MorePage extends StatelessWidget {
  const MorePage({super.key});
  @override
  Widget build(BuildContext c) => Page('More Insights', [
    const Text('Go deeper into your estate data', style: TextStyle(color: C.muted)),
    const V(18),
    _MoreTile(icon: Icons.trending_up_rounded, title: 'Plant Progress', subtitle: 'Track disease severity and team assignments', color: C.sand, tap: () => push(c, const PlantProgress())),
    const V(12),
    _MoreTile(icon: Icons.inventory_2_outlined, title: 'Resource Plan', subtitle: 'Prioritize labour, treatments and budget', color: C.mint, tap: () => push(c, const ResourcePlan())),
    const V(26),
    const Box(Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('FIELD NOTE', style: TextStyle(color: C.green, fontSize: 10, letterSpacing: 1.4, fontWeight: FontWeight.w800)),
      V(8),
      Text('Your next best action', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800)),
      V(7),
      Text('Inspect Block B4 before the next humidity spike. The team is ready to deploy.'),
    ],), color: C.white),
  ]);
}

class _MoreTile extends StatelessWidget {
  final IconData icon; final String title, subtitle; final Color color; final VoidCallback tap;
  const _MoreTile({required this.icon, required this.title, required this.subtitle, required this.color, required this.tap});
  @override
  Widget build(BuildContext context) => Box(Row(children: [
    Container(width: 50, height: 50, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(16)), child: Icon(icon, color: C.deep)),
    const H(14), Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(title, style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800)), SizedBox(height: 4), Text(subtitle, style: TextStyle(color: C.muted, fontSize: 12))])),
    const Icon(Icons.arrow_forward_rounded, color: C.green),
  ]), color: C.white, tap: tap);
}

Widget metric(String a, String b) => Expanded(
  child: Box(
    Column(
      children: [
        Text(
          a,
          style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
        ),
        const V(6),
        Text(b, textAlign: TextAlign.center),
      ],
    ),
    color: C.white,
  ),
);

class FarmOverview extends StatelessWidget {
  const FarmOverview({super.key});
  @override
  Widget build(BuildContext c) => Page('Farm Overview', [
    _HeroHealth(),
    const V(16),
    Row(
      children: [
        metric('4', 'Diseased'),
        const H(10),
        metric('2', 'High Risk'),
        const H(10),
        metric('12', 'Total Blocks'),
      ],
    ),
    const V(16),
    const Box(
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Section('⚡ UPCOMING RISKS', trailing: Text('2 alerts', style: TextStyle(color: C.red, fontWeight: FontWeight.bold))),
          V(12),
          Text('7-Day Severity Increase Forecast', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w800)),
          V(8),
          Text(
            'Block B4 shows early conditions for rapid pathogen spread due to microclimate changes.',
          ),
        ],
      ),
      color: C.sand,
    ),
    const V(16),
    ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: Image.asset(
        'assets/images/farm_map.png',
        height: 170,
        width: double.infinity,
        fit: BoxFit.cover,
      ),
    ),
  ]);
}

class _HeroHealth extends StatelessWidget {
  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(22),
    decoration: BoxDecoration(
      gradient: const LinearGradient(colors: [C.deep, C.green], begin: Alignment.topLeft, end: Alignment.bottomRight),
      borderRadius: BorderRadius.circular(26),
      boxShadow: const [BoxShadow(color: Color(0x24092f25), blurRadius: 18, offset: Offset(0, 8))],
    ),
    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        const Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('ESTATE PULSE', style: TextStyle(color: C.lime, fontSize: 10, letterSpacing: 1.4, fontWeight: FontWeight.w800)),
          SizedBox(height: 7),
          Text('Healthy canopy\nacross the estate', style: TextStyle(color: Colors.white, fontSize: 22, height: 1.12, fontWeight: FontWeight.w800)),
        ])),
        SizedBox(width: 108, height: 108, child: Stack(alignment: Alignment.center, children: [
          CircularProgressIndicator(value: .88, strokeWidth: 11, color: C.lime, backgroundColor: Color(0x335f9b77)),
          const Text('88%', style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.w800)),
        ])),
      ]),
      const SizedBox(height: 18),
      const Row(children: [Icon(Icons.cloud_outlined, color: Color(0xffc6e0c6), size: 17), SizedBox(width: 6), Text('Humidity spikes expected in 48h', style: TextStyle(color: Color(0xffd2e4d1), fontSize: 13))]),
    ]),
  );
}

class FieldMap extends StatelessWidget {
  const FieldMap({super.key});
  @override
  Widget build(BuildContext c) => Page('Field Map', [
    const Box(
      Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text('🔴 Affected'),
          Text('🟠 High Risk'),
          Text('🟢 Healthy'),
        ],
      ),
    ),
    const V(20),
    GridView.count(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisCount: 3,
      mainAxisSpacing: 12,
      crossAxisSpacing: 12,
      children:
          [
                'A1\n98%',
                'A2\n☼',
                'A3\n92%',
                'B1\n⚠ RISK',
                'B2\n95%',
                'B3\n99%',
                'C1\n88%',
                'C2\n◉',
                'C3\n⚠ RISK',
              ]
              .map(
                (e) => Box(
                  Center(
                    child: Text(
                      e,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              )
              .toList(),
    ),
    const V(20),
    const Box(
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('✣ AI RECOMMENDATION'),
          V(8),
          Text(
            'Block A2',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          Text('Leaf Rust Detected', style: TextStyle(color: C.muted)),
          V(10),
          Text(
            'Initial signs of rust spotted in the north-east quadrant. Immediate fungicide application recommended within 48 hours.',
          ),
        ],
      ),
      color: C.white,
    ),
  ]);
}

class PlantScan extends StatefulWidget {
  const PlantScan({super.key});
  @override
  State<PlantScan> createState() => _PlantScanState();
}

class _PlantScanState extends State<PlantScan> {
  bool loading = false;
  ImageProvider? selectedImage;
  XFile? selectedFile;

  Future<void> chooseImage(ImageSource source) async {
    final picked = await ImagePicker().pickImage(source: source, imageQuality: 92);
    if (picked == null || !mounted) return;
    final imageBytes = await picked.readAsBytes();
    if (!mounted) return;
    setState(() {
      selectedFile = picked;
      selectedImage = MemoryImage(imageBytes);
    });
  }

  Future<void> submitScan() async {
    final picked = selectedFile;
    if (picked == null) return;
    setState(() => loading = true);
    try {
      final result = await Component1Api.scan(picked);
      if (!mounted) return;
      await Navigator.push(context, MaterialPageRoute(builder: (_) => DetectionResult(result: result)));
    } catch (error) {
      if (mounted) notice(context, error.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => loading = false);
    }
  }

  @override
  Widget build(BuildContext c) => Page('Plant Scan', [
    const Box(
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Section('Location', trailing: Text('● Edge Device: Online')),
          V(12),
          Text('Plantation'),
          V(5),
          Box(
            Section(
              'Kurunegala Estate',
              trailing: Icon(Icons.keyboard_arrow_down),
            ),
            color: C.white,
          ),
          V(10),
          Row(
            children: [
              Expanded(
                child: Text('Field\nField 02', style: TextStyle(height: 2)),
              ),
              Spacer(),
              Expanded(child: Text('Block\nB-12', style: TextStyle(height: 2))),
            ],
          ),
        ],
      ),
    ),
    TextButton.icon(
      onPressed: () => push(c, const ScanHistory()),
      icon: const Icon(Icons.history),
      label: const Text('View Scan History'),
    ),
    const V(8),
    ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: Stack(
        alignment: Alignment.bottomCenter,
        children: [
          SizedBox(
            height: 440,
            width: double.infinity,
            child: selectedImage == null
                ? const Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.camera_alt_outlined, size: 64, color: C.green),
                      V(14),
                      Text('Capture or choose a banana leaf image', style: TextStyle(fontWeight: FontWeight.w700)),
                      V(6),
                      Text('Your selected image will appear here', style: TextStyle(color: C.muted)),
                    ],
                  )
                : Image(image: selectedImage!, fit: BoxFit.cover),
          ),
        ],
      ),
    ),
    const V(10),
    if (selectedFile != null)
      Box(
        Row(
          children: [
            const Icon(Icons.check_circle, color: C.green),
            const H(10),
            const Expanded(
              child: Text('Image ready for AI analysis', style: TextStyle(fontWeight: FontWeight.w700)),
            ),
            FilledButton.icon(
              onPressed: loading ? null : submitScan,
              icon: const Icon(Icons.auto_awesome, size: 18),
              label: Text(loading ? 'Working…' : 'Scan'),
            ),
          ],
        ),
        color: C.mint,
      ),
    if (selectedFile == null)
      FilledButton.icon(
        onPressed: loading ? null : () => chooseImage(ImageSource.camera),
        icon: const Icon(Icons.camera_alt_outlined),
        label: const Text('Capture Leaf to Begin'),
      ),
    const V(10),
    OutlinedButton.icon(
      onPressed: loading ? null : () => chooseImage(ImageSource.gallery),
      icon: const Icon(Icons.photo_library_outlined),
      label: Text(selectedFile == null ? 'Choose from Gallery' : 'Choose a Different Image'),
    ),
  ]);
}

class LiveRisk extends StatelessWidget {
  const LiveRisk({super.key});
  @override
  Widget build(BuildContext c) => Page('Live Risk', [
    const Text('Live Monitoring'),
    const Text('● Last updated: Just now'),
    const V(22),
    const Box(
      Section(
        'Block A - North Sector',
        trailing: Icon(Icons.keyboard_arrow_down),
      ),
    ),
    const V(12),
    link(c, 'View Risk Contributors', const RiskContributors()),
    const V(10),
    link(c, 'Risk Forecast', const RiskForecast()),
    const V(10),
    link(c, 'Historical Trends', const HistoricalTrends()),
    const V(22),
    Row(
      children: [
        metric('29°C', 'TEMP\nOptimal range'),
        const H(12),
        metric('78%', 'HUMIDITY\nHigh humidity'),
      ],
    ),
    const V(14),
    const Box(
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Section(
            '☁ RAINFALL (24H)',
            trailing: Chip(label: Text('Normal'), side: BorderSide.none),
          ),
          V(16),
          Text('12 mm', style: TextStyle(fontSize: 22)),
        ],
      ),
    ),
    const V(14),
    const Box(
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Section('SOIL MOISTURE', trailing: Text('45%')),
          V(12),
          LinearProgressIndicator(
            value: .45,
            color: C.green,
            backgroundColor: C.white,
          ),
          V(8),
          Text('Dry              Optimal          Saturated'),
        ],
      ),
    ),
  ]);
}

Widget link(BuildContext c, String t, Widget p) => Box(
  Section(t, trailing: const Icon(Icons.chevron_right)),
  tap: () => push(c, p),
);

class PlantProgress extends StatelessWidget {
  const PlantProgress({super.key});
  @override
  Widget build(BuildContext c) => Page('Plant Progress', [
    const Box(Section('⌖ Block B4', trailing: Icon(Icons.search))),
    const V(16),
    Box(
      Row(
        children: [
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('● ALERT ACTIVE'),
                Text('Black Sigatoka Detected'),
                V(16),
                Text('Severity'),
                Text('42% Moderate', style: TextStyle(fontSize: 18)),
                LinearProgressIndicator(value: .42, color: C.red),
              ],
            ),
          ),
          const H(12),
          ClipRRect(
            borderRadius: BorderRadius.circular(14),
            child: Image.asset(
              'assets/images/disease_leaf.png',
              width: 96,
              height: 96,
              fit: BoxFit.cover,
            ),
          ),
        ],
      ),
    ),
    const V(14),
    Row(
      children: [
        metric('28.4°C', 'Temperature'),
        const H(12),
        metric('82%', 'Humidity'),
      ],
    ),
    const V(16),
    const Box(
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Section('Assigned Employees'),
          V(12),
          Text('👤  Kamal Perera\n      Field Supervisor'),
          V(12),
          Text('👤  Sunil Silva\n      Technician'),
          V(12),
          Text('👤  Nimali Cooray\n      Field Worker'),
        ],
      ),
      color: C.white,
    ),
    const V(18),
    const Section('Quick Access'),
    const V(10),
    link(c, 'Progression History', const ProgressionHistory()),
    const V(10),
    link(c, 'Severity Detail', const SeverityDetail()),
    const V(10),
    link(c, 'Future Forecast', const FutureForecast()),
  ]);
}

class ResourcePlan extends StatelessWidget {
  const ResourcePlan({super.key});
  @override
  Widget build(BuildContext c) => Page('Resource Plan', [
    const Box(
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Section(
            'Farm Overview',
            trailing: Icon(Icons.park_outlined, size: 38),
          ),
          Text('12 Total Blocks'),
          V(16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              Text('3\nHigh Priority', textAlign: TextAlign.center),
              Text('5\nMedium Priority', textAlign: TextAlign.center),
              Text('4\nLow Priority', textAlign: TextAlign.center),
            ],
          ),
          V(16),
          Text(
            'Prioritization based on detection, risk, and progression',
            textAlign: TextAlign.center,
          ),
        ],
      ),
    ),
    const V(18),
    const Section('Action Required', trailing: Text('View All →')),
    const V(10),
    const Box(
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Rank 1   Block B4'),
          V(8),
          Text('Severe risk detected. Immediate action recommended.'),
          V(8),
          Text('Sensor data anomalous', style: TextStyle(color: C.muted)),
        ],
      ),
    ),
    const V(12),
    const Box(
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Rank 2   Block A1'),
          V(8),
          Text('High risk of disease progression. Monitor closely.'),
          V(8),
          Text('Visual inspection needed', style: TextStyle(color: C.muted)),
        ],
      ),
    ),
    const V(18),
    link(c, 'Labour Allocation', const LabourAllocation()),
    const V(10),
    link(c, 'Intervention List', const InterventionList()),
    const V(10),
    link(c, 'Budget & Efficiency', const BudgetScreen()),
  ]);
}

class InfoScreen extends StatelessWidget {
  final String title, hero;
  final List<(String, String)> rows;
  final String? image;
  final ImageProvider? imageProvider;
  final List<Widget> extra;
  final bool modern;
  const InfoScreen(
    this.title,
    this.hero,
    this.rows, {
    super.key,
    this.image,
    this.imageProvider,
    this.extra = const [],
    this.modern = false,
  });

  Widget _heroCard() => Container(
    width: double.infinity,
    padding: const EdgeInsets.all(22),
    decoration: BoxDecoration(
      gradient: const LinearGradient(colors: [C.deep, C.green], begin: Alignment.topLeft, end: Alignment.bottomRight),
      borderRadius: BorderRadius.circular(24),
      boxShadow: const [BoxShadow(color: Color(0x22092f25), blurRadius: 18, offset: Offset(0, 8))],
    ),
    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      const Text('COMPONENT 1 • AI ANALYSIS', style: TextStyle(color: C.lime, fontSize: 10, letterSpacing: 1.5, fontWeight: FontWeight.w800)),
      const V(12),
      Text(hero, style: const TextStyle(color: Colors.white, fontSize: 25, height: 1.1, fontWeight: FontWeight.w800)),
      const V(18),
      Container(height: 7, decoration: BoxDecoration(color: const Color(0x44ffffff), borderRadius: BorderRadius.circular(8)), child: FractionallySizedBox(alignment: Alignment.centerLeft, widthFactor: .82, child: Container(decoration: BoxDecoration(color: C.lime, borderRadius: BorderRadius.circular(8))))),
    ]),
  );
  @override
  Widget build(BuildContext c) => Detail(title, [
    if (image != null || imageProvider != null)
      ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Image(
          image: imageProvider ?? AssetImage(image!),
          height: 220,
          width: double.infinity,
          fit: BoxFit.cover,
        ),
      ),
    if (image != null || imageProvider != null) const V(16),
    if (modern) _heroCard() else Box(
      Column(
        children: [
          Text(
            hero,
            textAlign: TextAlign.center,
            style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
          ),
          const V(10),
          const LinearProgressIndicator(
            value: .72,
            minHeight: 8,
            color: C.green,
            backgroundColor: C.white,
          ),
        ],
      ),
    ),
    const V(18),
    ...rows.expand(
      (r) => [
        Box(
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                r.$1,
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
              const V(6),
              Text(r.$2, style: const TextStyle(color: C.muted, fontSize: 15)),
            ],
          ),
          color: modern ? const Color(0xfff8fbf6) : C.white,
        ),
        const V(12),
      ],
    ),
    ...extra,
  ]);
}

class DetectionResult extends StatelessWidget {
  final ScanResult result;
  const DetectionResult({super.key, required this.result});
  @override
  Widget build(BuildContext c) {
    final rejected = result.predictedClass == null;
    final scores = result.classScores.entries.map((entry) => '${displayDisease(entry.key)}: ${((entry.value as num) * 100).toStringAsFixed(1)}%').join('\n');
    return InfoScreen(
      'Detection Result',
      rejected ? (result.status == 'gatekeeper_uncertain' ? 'IMAGE UNCERTAIN' : 'IMAGE REJECTED') : 'PREDICTED\n${displayDisease(result.predictedClass!)}',
      [
        ('Status', result.status.replaceAll('_', ' ').toUpperCase()),
        if (result.confidence != null) ('Model Confidence', '${result.confidence!.toStringAsFixed(2)}%'),
        if (scores.isNotEmpty) ('Class Scores', scores),
        if (result.message != null) ('Message', result.message!),
      ],
      extra: [
        if (!rejected && result.overlayImage != null) Primary('View Explainable AI', () => push(c, XaiScreen(result: result))),
      ],
      imageProvider: base64Image(result.originalImage),
      modern: true,
    );
  }
}

class XaiScreen extends StatelessWidget {
  final ScanResult result;
  const XaiScreen({super.key, required this.result});
  @override
  Widget build(BuildContext c) => InfoScreen(
    'Explainable AI',
    '${result.explanationTarget == null ? 'Prediction' : displayDisease(result.explanationTarget!)} • ${result.confidence?.toStringAsFixed(2) ?? '—'}% Confidence',
    const [('Grad-CAM Explanation', 'Highlighted regions show the leaf areas that most influenced the prototype classification.')],
    imageProvider: base64Image(result.overlayImage),
    modern: true,
  );
}

class ScanHistory extends StatefulWidget {
  const ScanHistory({super.key});
  @override
  State<ScanHistory> createState() => _ScanHistoryState();
}

class _ScanHistoryState extends State<ScanHistory> {
  late Future<List<ScanResult>> scans = Component1Api.history();
  @override
  Widget build(BuildContext c) => FutureBuilder<List<ScanResult>>(
    future: scans,
    builder: (context, snapshot) {
      if (snapshot.connectionState == ConnectionState.waiting) return const Center(child: CircularProgressIndicator());
      if (snapshot.hasError) return InfoScreen('Scan History', 'Unable to connect', [('Error', snapshot.error.toString())]);
      final rows = snapshot.data ?? [];
      return InfoScreen('Scan History', '${rows.length} saved scans', rows.isEmpty
          ? const [('No scans yet', 'Run a banana-leaf scan to create history.')]
          : rows.map((scan) => (displayDisease(scan.predictedClass ?? 'Rejected'), '${scan.confidence?.toStringAsFixed(1) ?? '—'}% • ${scan.status.replaceAll('_', ' ')}')).toList(),
        modern: true,
      );
    },
  );
}

class RiskContributors extends StatelessWidget {
  const RiskContributors({super.key});
  @override
  Widget build(BuildContext c) => const InfoScreen(
    'Risk Contributors',
    'Risk Factors',
    [
      ('Humidity (88%)', '+25% Risk Contribution • Above Threshold • 70% Max'),
      ('Leaf Wetness', 'Critical Level • High'),
      ('Temp (30°C)', 'Optimal for Fungal Growth • Danger Zone 25–32°C'),
    ],
  );
}

class RiskForecast extends StatelessWidget {
  const RiskForecast({super.key});
  @override
  Widget build(BuildContext c) =>
      const InfoScreen('72-Hour Risk Forecast', '72% HIGH RISK', [
        ('AI FORECAST', 'High Risk of Panama Disease'),
        (
          'BLOCK INFORMATION',
          'Block B - North Sector • 12.5 Hectares • Cavendish G9',
        ),
        (
          'Recommended Immediate Action',
          'Initiate strict quarantine protocols for Sector 4 immediately. Apply preemptive fungicide treatment to adjacent sectors.',
        ),
      ], image: 'assets/images/risk_map.png');
}

class HistoricalTrends extends StatelessWidget {
  const HistoricalTrends({super.key});
  @override
  Widget build(BuildContext c) =>
      const InfoScreen('Historical Trends', 'Environmental Trends • 7d', [
        (
          'Temperature & Humidity',
          '7-Day Averages • Live\nMon     Wed     Fri     Sun',
        ),
        ('Rainfall', 'Last 7 Days • 12.4 mm'),
        ('Crop Risk Index', 'Stress Potential • Elevated'),
      ]);
}

class ProgressionHistory extends StatelessWidget {
  const ProgressionHistory({super.key});
  @override
  Widget build(BuildContext c) =>
      const InfoScreen('Progression History', 'Disease Timeline', [
        ('Oct 14 • MODERATE', '42% Severity • 24°C • 82% RH • Heavy Rain'),
        ('Oct 10 • MILD', '28% Severity • 22°C • 65% RH • Overcast'),
        ('Oct 05 • HEALTHY', '5% Severity • 26°C • 55% RH • Clear skies'),
      ], image: 'assets/images/timeline_1.png');
}

class SeverityDetail extends StatelessWidget {
  const SeverityDetail({super.key});
  @override
  Widget build(BuildContext c) =>
      const InfoScreen('Severity Details', 'SEVERITY SCORE • 42% Moderate', [
        ('Visual Evidence', 'Affected Area • Est. 15% of Total Canopy'),
        ('Microclimate', 'High Humidity'),
        ('Spread Rate', 'Accelerating'),
      ], image: 'assets/images/severity_leaf.png');
}

class FutureForecast extends StatelessWidget {
  const FutureForecast({super.key});
  @override
  Widget build(BuildContext c) =>
      const InfoScreen('Future Forecast', 'Disease Forecast', [
        ('Current 42%', '7 Days 55% • 14 Days 71%'),
        ('Rising Humidity', 'Accelerates spore growth'),
        ('Untreated Status', 'No intervention recorded'),
      ]);
}

class LabourAllocation extends StatelessWidget {
  const LabourAllocation({super.key});
  @override
  Widget build(BuildContext c) => const InfoScreen(
    'Labour Allocation',
    'Workers Available • 15 / 20 • 75%',
    [
      (
        'Active Allocations',
        'Block B4 • Spraying & Inspection • 4 Workers • Allocated',
      ),
      (
        'Pending Allocations',
        'Field 02 • Containment Zone • 2 Workers • Pending',
      ),
    ],
  );
}

class InterventionList extends StatelessWidget {
  const InterventionList({super.key});
  @override
  Widget build(BuildContext c) =>
      const InfoScreen('Interventions', 'Intervention Priority', [
        (
          'BLOCK B4 • Sigatoka Disease',
          'CRITICAL • Severity 42% • Fungicide Application',
        ),
        (
          'BLOCK A1 • High Risk Zone',
          'HIGH • Risk Level Elevated • Targeted Monitoring',
        ),
      ]);
}

class BudgetScreen extends StatelessWidget {
  const BudgetScreen({super.key});
  @override
  Widget build(BuildContext c) => const InfoScreen(
    'Budget & Efficiency',
    'Budget Overview • 24% Utilized',
    [
      ('Total Spent Rs 1,200', 'Budget Rs 5,000 • On track'),
      ('Cost Breakdown', 'Labour Rs 600 • Treatment Rs 450 • Transport Rs 150'),
      (
        'Optimization Score 92%',
        'Excellent utilization. Rs 150 saved via prioritized allocation this week.',
      ),
      (
        'AI Savings Opportunities',
        'Combine Transport Routes • Est. Rs 80\nReduce Irrigation Sector A • Est. Rs 45',
      ),
    ],
  );
}
