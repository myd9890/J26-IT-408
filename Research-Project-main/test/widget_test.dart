import 'package:cavendish/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('shows AgriLeaf splash and login', (tester) async {
    await tester.pumpWidget(const AgriLeafApp());
    expect(find.text('AgriLeaf AI'), findsOneWidget);
    expect(find.text('INITIALIZING SENSORS...'), findsOneWidget);
    await tester.pump(const Duration(milliseconds: 1700));
    await tester.pumpAndSettle();
    expect(find.text('Email or Worker ID'), findsOneWidget);
  });
}
