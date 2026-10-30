import 'package:flutter/material.dart';
import 'package:battery_plus/battery_plus.dart';
import '../theme/app_theme.dart';
import '../theme/app_widgets.dart';
import 'home_screen.dart';

class BatteryScreen extends StatefulWidget {
  const BatteryScreen({super.key});

  @override
  State<BatteryScreen> createState() => _BatteryScreenState();
}

class _BatteryScreenState extends State<BatteryScreen> {
  final Battery _battery = Battery();

  int _phoneBattery = 0;
  BatteryState _phoneState = BatteryState.unknown;

  // ignore: prefer_final_fields
  int _glassesBattery = 0;
  // ignore: prefer_final_fields
  bool _glassesConnected = false;

  // ignore: prefer_final_fields
  String _phoneEstimate = "4h 30m";
  // ignore: prefer_final_fields
  String _glassesEstimate = "2h 15m";

  @override
  void initState() {
    super.initState();
    _loadPhoneBattery();
    _listenPhoneBatteryChanges();
  }

  Future<void> _loadPhoneBattery() async {
    final level = await _battery.batteryLevel;
    final state = await _battery.batteryState;
    if (!mounted) return;
    setState(() {
      _phoneBattery = level;
      _phoneState = state;
    });
  }

  void _listenPhoneBatteryChanges() {
    _battery.onBatteryStateChanged.listen((state) {
      if (mounted) setState(() => _phoneState = state);
    });
  }

  String _getStatusText(BatteryState state) {
    switch (state) {
      case BatteryState.charging:
        return "Charging";
      case BatteryState.discharging:
        return "Active";
      case BatteryState.full:
        return "Full";
      default:
        return "Unknown";
    }
  }

  void _goBackToHome() {
    final homeState = context.findAncestorStateOfType<HomeScreenState>();
    if (homeState != null) {
      homeState.changeTab(0);
    } else {
      Navigator.maybePop(context);
    }
  }

  Color _batteryColor(int level) {
    if (level > 50) return AppColors.success;
    if (level > 20) return AppColors.warning;
    return AppColors.danger;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text("Battery Status"),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: _goBackToHome,
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_rounded),
            tooltip: "Refresh",
            onPressed: _loadPhoneBattery,
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(AppSpacing.md),
        child: Column(
          children: [
            _buildBatteryCard(
              title: "Smart Glasses",
              device: "ESP32-CAM Device",
              batteryLevel: _glassesBattery,
              estimatedTime: _glassesEstimate,
              status: _glassesConnected ? "Active" : "Inactive",
              isGlasses: true,
              connected: _glassesConnected,
            ),
            const SizedBox(height: AppSpacing.md),
            _buildBatteryCard(
              title: "Smartphone",
              device: "Android Device",
              batteryLevel: _phoneBattery,
              estimatedTime: _phoneEstimate,
              status: _getStatusText(_phoneState),
              isGlasses: false,
              connected: true,
            ),
            const SizedBox(height: AppSpacing.lg),
          ],
        ),
      ),
    );
  }

  Widget _buildBatteryCard({
    required String title,
    required String device,
    required int batteryLevel,
    required String estimatedTime,
    required String status,
    required bool isGlasses,
    required bool connected,
  }) {
    final color = _batteryColor(batteryLevel);
    final charging = status == "Charging";

    return AppCard(
      padding: const EdgeInsets.all(AppSpacing.lg),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header row
          Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: AppDecorations.pill(AppColors.primary),
                child: Icon(
                  isGlasses ? Icons.smart_toy_rounded : Icons.smartphone_rounded,
                  color: AppColors.primary,
                  size: 22,
                ),
              ),
              const SizedBox(width: AppSpacing.sm),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: AppText.subheading),
                    Text(device, style: AppText.label),
                  ],
                ),
              ),
              AppStatusPill(
                text: status,
                color: charging
                    ? AppColors.success
                    : (connected ? AppColors.primary : AppColors.textMuted),
                icon: charging ? Icons.bolt_rounded : null,
              ),
            ],
          ),
          const SizedBox(height: AppSpacing.lg),

          // Battery bar + big percent
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Text(
                "$batteryLevel",
                style: TextStyle(
                  color: color,
                  fontSize: 46,
                  fontWeight: FontWeight.bold,
                  height: 1,
                ),
              ),
              const Padding(
                padding: EdgeInsets.only(top: 8, left: 2),
                child: Text("%",
                    style: TextStyle(
                        color: AppColors.textSecondary,
                        fontSize: 20,
                        fontWeight: FontWeight.bold)),
              ),
              const Spacer(),
              if (charging)
                const Icon(Icons.bolt_rounded,
                    color: AppColors.success, size: 28),
            ],
          ),
          const SizedBox(height: AppSpacing.sm),

          // Progress bar
          ClipRRect(
            borderRadius: BorderRadius.circular(AppRadius.pill),
            child: LinearProgressIndicator(
              value: batteryLevel / 100,
              minHeight: 10,
              backgroundColor: AppColors.background,
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
          const SizedBox(height: AppSpacing.md),

          // Estimated time row
          Row(
            children: [
              const Icon(Icons.access_time_rounded,
                  color: AppColors.textMuted, size: 18),
              const SizedBox(width: 8),
              const Text("Estimated Time", style: AppText.bodySecondary),
              const Spacer(),
              Text(
                connected ? estimatedTime : "—",
                style: const TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
