/**
 * Angel iOS / React Native — camera vision with Describe vs Forensic modes.
 * Point API_BASE at your deployed Angel URL (same host as /api/vision).
 */
import React, { useCallback, useState } from "react";
import {
  View,
  Text,
  Button,
  ScrollView,
  TextInput,
  Switch,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from "react-native";
import * as ImagePicker from "expo-image-picker";

const API_BASE = process.env.EXPO_PUBLIC_ANGEL_API || "https://your-angel-host";

export default function App() {
  const [forensicMode, setForensicMode] = useState(false);
  const [context, setContext] = useState("");
  const [tylerLocation, setTylerLocation] = useState("");
  const [loading, setLoading] = useState(false);
  const [describeReply, setDescribeReply] = useState("");
  const [forensic, setForensic] = useState(null);
  const [lastImageB64, setLastImageB64] = useState("");

  const runVision = useCallback(async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert("Camera permission required");
      return;
    }
    const shot = await ImagePicker.launchCameraAsync({
      base64: true,
      quality: 0.85,
    });
    if (shot.canceled || !shot.assets?.[0]) return;
    const asset = shot.assets[0];
    const b64 = asset.base64;
    if (!b64) {
      Alert.alert("Camera", "Enable base64 on the image picker (base64: true).");
      setLoading(false);
      return;
    }
    setLastImageB64(b64);
    setLoading(true);
    setDescribeReply("");
    setForensic(null);

    try {
      if (!forensicMode) {
        const q =
          (context || "").trim() ||
          "Describe this image in detail for Tyler.";
        const r = await fetch(`${API_BASE}/api/vision`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: b64,
            question: q,
            device: "ios",
          }),
        });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || r.statusText);
        setDescribeReply(j.reply || "");
      } else {
        const r = await fetch(`${API_BASE}/api/vision/forensic`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_base64: b64,
            context: context || "Forensic visual analysis.",
            tyler_location: tylerLocation || undefined,
            file_name: "ios-camera.jpg",
          }),
        });
        const j = await r.json();
        if (!r.ok || j.ok === false) throw new Error(j.error || r.statusText);
        setForensic(j);
      }
    } catch (e) {
      Alert.alert("Vision error", String(e.message || e));
    } finally {
      setLoading(false);
    }
  }, [forensicMode, context, tylerLocation]);

  const fileToIntel = useCallback(async () => {
    if (!lastImageB64 || !forensic) return;
    setLoading(true);
    try {
      const payload = { ...forensic };
      delete payload.mission_cross_reference;
      delete payload.network_updates_applied;
      const r = await fetch(`${API_BASE}/api/vision/forensic/file`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: lastImageB64,
          forensic_json: payload,
          file_name: "ios-camera.jpg",
        }),
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || "file failed");
      Alert.alert("Filed", j.cabinet_file || "Visual Intelligence");
      setForensic((prev) =>
        prev ? { ...prev, show_manual_file_button: false, auto_filed: true } : prev
      );
    } catch (e) {
      Alert.alert("File error", String(e.message || e));
    } finally {
      setLoading(false);
    }
  }, [lastImageB64, forensic]);

  const primaryType =
    forensic?.classification?.primary_type ||
    forensic?.analysis_type ||
    "—";

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Angel Vision</Text>

      <View style={styles.row}>
        <Text>Describe</Text>
        <Switch
          value={forensicMode}
          onValueChange={setForensicMode}
          trackColor={{ true: "#6a9cff" }}
        />
        <Text>Forensic</Text>
      </View>

      <TextInput
        style={styles.input}
        placeholder="Context / question"
        value={context}
        onChangeText={setContext}
        multiline
      />
      {forensicMode ? (
        <TextInput
          style={styles.input}
          placeholder="Tyler location (optional)"
          value={tylerLocation}
          onChangeText={setTylerLocation}
        />
      ) : null}

      <Button
        title={loading ? "…" : "Take photo & analyze"}
        onPress={runVision}
        disabled={loading}
      />
      {loading ? <ActivityIndicator style={{ marginTop: 16 }} /> : null}

      {!forensicMode && describeReply ? (
        <Text style={styles.body}>{describeReply}</Text>
      ) : null}

      {forensicMode && forensic ? (
        <View style={{ marginTop: 16 }}>
          <Text style={styles.badge}>TYPE: {String(primaryType).toUpperCase()}</Text>
          <Text style={styles.badge}>
            CONFIDENCE: {forensic.confidence || "—"}
          </Text>
          <Text style={styles.badge}>
            MISSION: {forensic.mission_relevance || "—"}
          </Text>
          <Text style={styles.summary}>
            {forensic.summary_for_progressive_ui || forensic.summary}
          </Text>
          <Text style={styles.section}>Key findings</Text>
          {(forensic.key_findings || []).map((k, i) => (
            <Text key={i} style={styles.bullet}>
              • {k}
            </Text>
          ))}
          <Text style={styles.section}>Anomalies</Text>
          {(forensic.anomalies || []).length ? (
            (forensic.anomalies || []).map((a, i) => (
              <Text key={i} style={styles.anomaly}>
                ⚠ {a}
              </Text>
            ))
          ) : (
            <Text style={styles.muted}>None noted</Text>
          )}
          <Text style={styles.action}>{forensic.recommended_action}</Text>
          {forensic.show_manual_file_button ? (
            <Button title="File to Intelligence" onPress={fileToIntel} />
          ) : null}
          {forensic.auto_filed ? (
            <Text style={styles.muted}>Auto-filed to Visual Intelligence</Text>
          ) : null}
        </View>
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, paddingTop: 56 },
  title: { fontSize: 22, fontWeight: "700", marginBottom: 12 },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    marginBottom: 12,
  },
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 8,
    padding: 10,
    marginBottom: 10,
    minHeight: 44,
  },
  body: { marginTop: 16, fontSize: 15, lineHeight: 22 },
  badge: {
    fontSize: 12,
    fontWeight: "600",
    color: "#1a5fb4",
    marginBottom: 4,
  },
  summary: { fontSize: 16, marginVertical: 10, lineHeight: 24 },
  section: { fontWeight: "700", marginTop: 12 },
  bullet: { fontSize: 14, marginLeft: 8, marginTop: 4 },
  anomaly: { fontSize: 14, color: "#b35900", marginTop: 4 },
  action: { marginTop: 12, fontSize: 14, fontStyle: "italic" },
  muted: { marginTop: 8, color: "#666", fontSize: 13 },
});
